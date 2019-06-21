import sys
import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import DRIT
from saver import Saver


def main():

    debug_mode=False

    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # daita loader
    print('\n--- load dataset ---')
    dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)
    '''
        通过检查dataset_unpair，我们发现：
            图像是先缩放到256,256,然后再随机裁剪出216,216的patch，（测试时是从中心裁剪）
    '''

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    if not debug_mode:
        model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        '''
            images_a,images_b: 2,3,216,216
        '''
        for it, (images_a, images_b) in enumerate(train_loader):
            #   假如正好拿到了残次的剩余的一两个样本，就跳过，重新取样
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue

            # input data
            if not debug_mode:
                images_a = images_a.cuda(opts.gpu).detach() #   这里进行detach，可能是为了避免计算不需要的梯度，节省显存
                images_b = images_b.cuda(opts.gpu).detach()

            # update model 按照默认设置，1/3的iter更新内容判别器，2/3的iter更新D和EG
            if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images_a, images_b)
                continue
            else:
                model.update_D(images_a, images_b)
                model.update_EG()

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            sys.stdout.flush()
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, total_it, model)

    return


if __name__ == '__main__':
    main()

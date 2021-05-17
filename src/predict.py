import os
import time

from .lib import *


# path = base + f"data/washer/washer_ng/{folder}"
def predict(
    path: str, weight_path: str, threshold: float, 
    num_workers: int = 2, batch_size: int = 4, 
    show_images: bool = True, 
    save_images: bool = True,
    fname=''):
    """
    # Input
    - path: path of the image or directory of images to be tested
    - weight_path: path of the model weight file
    - num_workers: number of workers to load data
    - batch_size: batch size for the model to pridict
    - show_images: If True, it shows the image of the heatmap
    """
    img_size = 128
    mean = std = [0.5]*3
    aug_seq = augment(img_size=img_size, mean=mean, std=std)
    # threshold = 0.008411495946347713
    # threshold = 0.012
    if os.path.isfile(path):
        dataset = MyDataset([path], "", mode="test",
                            aug=aug_seq,
                            test_label=0, img_size=img_size)
        testloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_len = 1
        batch_size = 1
    elif os.path.isdir(path):
        testloader, test_len = get_dataloader(
            path, mode='test', batch_size=batch_size, shuffle=False, num_workers=num_workers, img_size=img_size, mean=mean, std=std)
    device = 'cuda'
    model = Autoencoder(img_size).to(device)
    model.load_state_dict(torch.load(weight_path))
    dataiter = iter(testloader)
    all_labels = []
    for i in range(int(np.ceil(test_len/batch_size))):
        images, labels = dataiter.next()
        images = images.to(device)
        output = model(images)
        images = images.cpu().numpy()
        output = output.view(-1, 3, img_size, img_size)
        output = output.detach().cpu().numpy()

        fscore = feature_score(images, output)
        tp = 100
        iscore = instance_score(fscore, tp)
        score = iscore

        # print(f'fscore:{fscore}')
        # print(f'iscore:{iscore}')
        # print(f'iscore range:{min(iscore)}, {max(iscore)}')
        # print(f'threshold:{threshold}')
        labels = (score > threshold).astype(int)
        all_labels += list(labels)
        print(f'iscore:{[round(s, 3) for s in iscore]}\tthreshold:{round(threshold, 3)}\tAnamoly:{bool(labels)}')
        # print(f'labels:{labels}')
        # print(f'total:{sum(labels)}/{len(labels)}')
        # print(f'percent:{round(sum(labels)/len(labels)*100, 3)}%')
        
    print(
        f"::::::::::::::::::::::{path.split('/')[-1]}:::::::::::::::::::::::::")
    # result[folder] = round(sum(all_labels)/len(all_labels)*100, 3)
    print(f'Total anomaly found:{sum(all_labels)}/{len(all_labels)}')
    # print(f'percent:{result[folder]}%')

    if show_images:
        ncols = 3
        fig, axes = plt.subplots(nrows=batch_size, ncols=ncols,
                                 sharex=True, sharey=True, figsize=(2*ncols, 2*batch_size))
        i = 0
        if batch_size > 1:
            axes[0, 0].set_title('image')
            axes[0, 1].set_title('output')
            axes[0, 2].set_title('heatmap')
            for img, ax in zip(images, axes[:, i]):
                ax.imshow(tensor2np(img))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            for img, ax in zip(output, axes[:, i]):
                ax.imshow(tensor2np(img))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            # for img, ax in zip(fscore, axes[:, i]):
            #     ax.imshow(tensor2np(img))
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
            for img, ax in zip(fscore, axes[:, 2]):
                im = tensor2np(img)
                ax.imshow(np.dot(im[...,:3], [0.2989, 0.5870, 0.1140]))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            # for img, ax in zip(fscore, axes[:, i]):
            #     ax.imshow(tensor2np(img)[:, :, 1])
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
            # for img, ax in zip(fscore, axes[:, i]):
            #     ax.imshow(tensor2np(img)[:, :, 2])
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
        elif batch_size == 1:
            axes[0].set_title('image')
            axes[1].set_title('output')
            for img, ax in zip(images, [axes[i]]):
                ax.imshow(tensor2np(img))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            for img, ax in zip(output, [axes[i]]):
                ax.imshow(tensor2np(img))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            # for img, ax in zip(fscore, [axes[i]]):
            #     ax.imshow(tensor2np(img))
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
            for img, ax in zip(fscore, [axes[i]]):
                im = tensor2np(img)
                ax.imshow(np.dot(im[...,:3], [0.2989, 0.5870, 0.1140]))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            # for img, ax in zip(fscore, [axes[i]]):
            #     ax.imshow(tensor2np(img)[:, :, 1])
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
            # for img, ax in zip(fscore, [axes[i]]):
            #     ax.imshow(tensor2np(img)[:, :, 2])
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            #     i += 1
            if save_images:
                i = 0
                if os.path.exists(fname + f"/{i}.png"):
                    i += 1
                filename = fname + f"/{i}.png"
                fig.savefig(
                    filename, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
            else:
                plt.show()
                
    
    return fig


if __name__ == "__main__":
    tic = time.process_time()
    # path = "data/washer/washer_ng/kizu"
    # path = "data/washer/washer_ng/sabi"
    # path = "data/washer/washer_ng/sure"
    # path = "data/washer/washer_ok"
    # path = "/home/jitesh/jg/washer/data0/washer_ng/sabi/WIN_20200505_13_22_47_Pro.jpg"
    paths = [
        "data/washer/washer_ng/kizu",
        "data/washer/washer_ng/sabi",
        "data/washer/washer_ng/sure",
        "data/washer/washer_ok",
        "data/washer/washer_ng/sabi/WIN_20200505_13_22_47_Pro.jpg"
             ]
    for path in paths:
        print(path)
        predict(
            path=path,
            weight_path="/home/jitesh/jg/anomaly_detection/weights/simple2/1.pth",
            threshold = 0.008411495946347713,
            batch_size=1,
            # show_images=False,
            )
        toc = time.process_time()
    print(f'Time taken for inference: {(toc - tic)}')

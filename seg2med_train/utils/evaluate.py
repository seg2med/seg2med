import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seg2med_train.utils.image_metrics import ImageMetrics

def calculate_mask_metrices(val_output, val_labels, val_masks,
                            log_file_overall, val_step, dynamic_range = [-1024., 3000.], printoutput=False):
    metricsCalc=ImageMetrics(dynamic_range)

    if val_masks is None:
        val_ssim = metricsCalc.ssim(val_output.numpy(), val_labels.numpy()) # 
        val_mae = metricsCalc.mae(val_output.numpy(), val_labels.numpy())
        val_psnr = metricsCalc.psnr(val_output.numpy(), val_labels.numpy())
    else:
        val_ssim = metricsCalc.ssim(val_output.numpy(), val_labels.numpy(), val_masks.numpy()) # 
        val_mae = metricsCalc.mae(val_output.numpy(), val_labels.numpy(), val_masks.numpy())
        val_psnr = metricsCalc.psnr(val_output.numpy(), val_labels.numpy(), val_masks.numpy())

    val_metrices = {
        'ssim': val_ssim,
        'mae': val_mae,
        'psnr': val_psnr,
    }

    if printoutput:
        print(f"mean ssim {val_step}: {val_metrices['ssim']}") #:.4f
        print(f"mean mae {val_step}: {val_metrices['mae']}")
        print(f"mean psnr {val_step}: {val_metrices['psnr']}")

    #output_val_log('mean', val_step, val_log_file=log_file_overall, val_metrices=val_metrices)
    ssim = val_metrices.get('ssim', 0)
    mae = val_metrices.get('mae', 0)
    psnr = val_metrices.get('psnr', 0)
    with open(log_file_overall, 'a') as f:
        f.write(f'mean metrics {val_step}, SSIM: {ssim}, MAE: {mae}, PSNR: {psnr}\n')
    return val_metrices


    


   


# Define function to reverse normalization
def reverse_normalize_data(tensor, 
                           mean=None, 
                           std=None, 
                           min_val=None, 
                           max_val=None, 
                           mode='zscore'):
    if mode == 'zscore':
        return tensor * std + mean if mean is not None and std is not None else tensor
    elif mode == 'minmax':
        return (tensor+1) /2 * (max_val - min_val) + min_val if min_val is not None and max_val is not None else tensor
    elif mode == 'inputonlyminmax' or mode == 'none' or mode == 'inputonlyzscore':
        return tensor
    elif mode == 'scale1000':
        return tensor * 1000-1024
    elif mode == 'scale4000':
        return tensor * 4000-1024
    elif mode == 'scale2000':
        return tensor * 2000-1000
    elif mode == 'nonegative':
        return tensor - 1024
    elif mode == 'norm_mr':
        return tensor*255
    elif mode == 'norm_mr_scale':
        return tensor*255


# Define function to save images
def save_single_image(input_imgs,filename, imgformat, dpi=300):
    plt.figure() #, figsize=(5, 4))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(input_imgs, cmap='gray')
    plt.savefig(filename, format=f'{imgformat}'
                , bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

        # Continue with arranging and saving the images as before, but use the above methods

def arrange_images(input_imgs, 
                   label_imgs,
                   fake_imgs,
                   model_name,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        titles = ['MRI', 'CT', model_name]
        fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        cnt = 0

        #print(gen_imgs[cnt].shape)
        axs[0].imshow(input_imgs, cmap='gray') # 0,0,
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(label_imgs, cmap='gray')
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        axs[2].imshow(fake_imgs, cmap='gray')
        axs[2].set_title(titles[2])
        axs[2].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

# Define function to plot histograms
def plot_histogram(data, title, ax, color='blue', alpha=0.7, 
                   x_lower_limit=-1, x_upper_limit=3, y_lower_limit=0, y_upper_limit=15000):
    #x_lower_limit, x_upper_limit = -100, 300 #-1100, 3000
    #y_lower_limit, y_upper_limit = 0, 15000
    bins = 256
    ax.hist(data.flatten(), bins=bins,range=(x_lower_limit, x_upper_limit), color=color, alpha=alpha)
    ax.set_ylim([y_lower_limit, y_upper_limit])
    ax.set_title(title)
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Frequency')

    

# Arrange three histograms
def arrange_3_histograms(source, target, output, saved_name , dpi=300,
                         x_lower_limit=-1, x_upper_limit=3, y_lower_limit=0, y_upper_limit=15000):
    # Plot histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plot_histogram(source, f'Histogram for source', axs[0],color='red',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    plot_histogram(target, f'Histogram for target', axs[1],color='green',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    plot_histogram(output, f'Histogram for output', axs[2],color='blue',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    #plot_histogram(transformed, f'Histogram for transformed {mode}', axs[2],color='blue')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name, dpi=dpi)
    plt.close(fig)

    # boxplot
    data = [source.flatten(), target.flatten(), output.flatten()]
    plt.boxplot(data, autorange = True)
    plt.xticks([1, 2, 3], ['Source', 'Target', 'Fake'])
    plt.title('Pixel Value Distribution')
    plt.xlabel('Image Type')
    plt.ylabel('Pixel Values')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name.replace('histogram','boxplot'), dpi=dpi)
    plt.close()


 
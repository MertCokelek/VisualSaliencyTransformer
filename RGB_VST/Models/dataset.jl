export load_data, preprocess_img

using Pkg, Colors, Images, Knet, Statistics, Plots; default(fmt = :png) 
using Base.Iterators: flatten
using IterTools: ncycle, takenth


train_image_paths = split(read(`ls -1v /home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Image`, String));
train_mask_paths = split(read(`ls -1v /home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Mask`, String));
train_contour_paths = split(read(`ls -1v /home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Mask`, String));

train_image_paths = ["/home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Image/"*i for i in train_image_paths]
train_mask_paths = ["/home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Mask/"*i for i in train_mask_paths];
train_contour_paths = ["/home/mcokelek21/Desktop/Github/VST/RGB_VST/Data/DUTS/DUTS-TR/DUTS-TR-Mask/"*i for i in train_contour_paths];


function preprocess_img(instance; imsize=224, random_crop=true, random_flip=true)
    new_img = deepcopy(instance[:,:,1]) # img = 0 at the end?
    new_msk = deepcopy(instance[:,:,2]) # img = 0 at the end?
    new_cnt = deepcopy(instance[:,:,3]) # img = 0 at the end?
    if random_flip
        if rand() > 0.5
            new_img = reverse(new_img, dims=2)
            new_msk = reverse(new_msk, dims=2)
            new_cnt = reverse(new_cnt, dims=2)
        end
    end
    
    if random_crop
        scale_size = 256
        x1 = rand(1: scale_size-imsize)
        y1 = rand(1: scale_size-imsize)
        new_img = new_img[x1:x1+imsize-1, y1:y1+imsize-1]
        new_msk = new_msk[x1:x1+imsize-1, y1:y1+imsize-1]
        new_cnt = new_cnt[x1:x1+imsize-1, y1:y1+imsize-1]
    end
    return new_img, new_msk, new_cnt
end


function load_data(img_paths, mask_paths, contour_paths; imsize=224, temp_n = 32)
    images = 0
    labels = 0
    contours = 0
    first = true
    temp_n_samples = 0
    for (path_img, path_msk, path_con) in zip(img_paths, mask_paths, contour_paths)
        if temp_n_samples < temp_n
            temp_n_samples += 1
            img = Images.imresize(load(path_img), 256, 256)
            label = Images.imresize(load(path_msk), 256, 256)
            contour = Images.imresize(load(path_con), 256, 256)
            
            if first
                images = cat(img, dims=4)
                labels = cat(label, dims=3)
                contours = cat(label, dims=3)
                first = false
            else
                images = cat(images, img, dims=4)
                labels = cat(labels, label, dims=3)
                contours = cat(contours, contour, dims=3)
            end
        end
    end
    # gt = permutedims(cat(labels, contours, dims=4), [1,2,4,3]) # GT's are concatenated. Permute required for Knet Minibatch
    gt = cat(labels, contours, dims=4) # GT's are concatenated. Permute required for Knet Minibatch
    return images, permutedims(gt, (1,2,4,3))
end;

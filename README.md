# SuperResolution
Training a network to upscale a downsampled image(with noise,blur,etc.)

#Steps to train your super-resolution network:
1. I have collected data from simulated environment but it can be any dataset(eg:DIV2K).
2. Designed the network architecture(used EDSR network and tweaked some of it's hyper-parameters).
3. Trained the network on GPU(Titan GTX 1080-Ti).
4. Now extract the common feature points between the reconstructed image and the ground truth image using ORB feature extractor.
5. I have used Mask-RCNN for semantic segmentation and have compared the feature points of reconstructed images and masked pixels for finding the missing objects for a network.

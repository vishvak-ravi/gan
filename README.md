# GAN
An implementation based on the original paper by Goodfellow et. al.

## Model Architecture Considerations
The models are constructed such that the generator uses a "mixture of rectifier linear activations and sigmoid activations" with the discriminator constructed with "maxout activations" and dropout. No additional information is given upon the implementation

### ZERO Variants
GeneratorZero and DiscriminatorZero are basic implementations true to the paper without using any additional features such as convolutional layers or residual connections. 

### General Concepts
Layer depths are kept small due to hardware limitations (MacBook Air M1 2020) but also unnecessary due to using low-resolution dataset MNIST.\

Layer widths were justified by initially expanding widths and subsequently shrinking to desired size ((28 x 28) and (1) for generator and discriminator respectively). Just heuristical here.



### 
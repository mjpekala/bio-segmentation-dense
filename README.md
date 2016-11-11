# Dense Image Segmentation


Some codes for segmenting images via per-pixel (or dense) classification.  The core idea is based on semantic segmentation [Long et al.] and we borrow the architecture from [Ronneberger et al.].

**Note: this code is under construction and subject to change.  Use at your own risk! ** 

## Quick Start

### Dimension Ordering

Note that this code makes explicit assumptions about the order of data dimensions (we expect tensors with shape (n_examples, n_channels, n_rows, n_cols)).  If you experience ugly-looking errors from underlying Thenao codes you may want to make sure that the Keras dimension ordering is set to "th"; e.g here is my keras.json:

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```


## References:

1.  Long, Shelhamer, Darrell "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.
2.  Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical Image Segmentation." 2015. https://arxiv.org/abs/1505.04597.
3.  Jocic, Keras implementation of U-Net. https://github.com/jocicmarko/ultrasound-nerve-segmentation.

# Train OCT CNNs
#
# mjp, may 2017

train :
	THEANO_FLAGS="device=gpu1,floatX=float32" python train_oct.py


clean :
	\rm -f oct_segmentation_*.{npz,hdf5}
	\rm -f oct_detection_*.{npz,hdf5}

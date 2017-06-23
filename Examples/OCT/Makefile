# Train OCT CNNs
#
# mjp, may 2017



train :
	THEANO_FLAGS="device=gpu1,floatX=float32" python train_oct.py



launch-fighters :
	nohup THEANO_FLAGS="device=gpu1,floatX=float32" python train_oct.py trial_A 10 1000 &
	sleep 5
	nohup THEANO_FLAGS="device=gpu2,floatX=float32" python train_oct.py trial_B 5 100 &
	sleep 5
	nohup THEANO_FLAGS="device=gpu3,floatX=float32" python train_oct.py trial_C 100 100 &
	sleep 5
	nohup THEANO_FLAGS="device=gpu4,floatX=float32" python train_oct.py trial_D 5 10 &
	sleep 5
	nohup THEANO_FLAGS="device=gpu5,floatX=float32" python train_oct.py trial_E 1 1000 &
	sleep 5
	nohup THEANO_FLAGS="device=gpu6,floatX=float32" python train_oct.py trial_F 20 1000 &



clean :
	\rm -f oct_segmentation_*.{npz,hdf5}
	\rm -f oct_detection_*.{npz,hdf5}

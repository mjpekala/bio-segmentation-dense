#
#  Demonstrates a quick training run on ISBI 2012.
#  I typically train on the cluster in the background.
#
#  Use the notebook to visualize the result after training is complete.
#

# mjp, 2017

train :
	nohup python train_isbi.py &


clean :
	\rm -f valid_epoch*.npz
	\rm -f weights_epoch*.hdf5
	\rm -f nohup.out

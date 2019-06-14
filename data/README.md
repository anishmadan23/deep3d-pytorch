# Handling KITTI RAW Data
The training/test data consists of stereo pairs from [KITTI RAW Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) 

**Train+Val Data** comprised of stereo pairs from following files:
- 2011_09_26_drive_0011_sync.zip
- 2011_09_26_drive_0022_sync.zip
- 2011_09_26_drive_0059_sync.zip
- 2011_09_26_drive_0084_sync.zip
- 2011_09_26_drive_0093_sync.zip
- 2011_09_26_drive_0095_sync.zip
- 2011_09_26_drive_0096_sync.zip

**Test Data** comprised of stereo pairs from following files:
- 2011_09_26_drive_0019_sync.zip
- 2011_09_26_drive_0091_sync.zip

## Creating train, val and test data
- Place all these files in this (data/) folder.  
**NOTE**: Remove the outermost folder after extracting each of the following zip files, i.e if 2011_09_26_drive_0096_sync.zip
is extracted, then outermost folder is 2011_09_26/. Remove this folder so that the outermost folder is 
now 2011_09_26_drive_0096_sync/.  

- Run make_train_test_data.py to get train,val and test data.
```
python3 make_train_test_data.py
```

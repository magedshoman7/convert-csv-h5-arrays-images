# convert-csv-h5-arrays-images
This code efficiently performs spatial and temporal binning of CSV data points into h5 arrays/images. Spatial binning is performed using latitude and longitude coordinates of data points. Temporal binning is performed using data points timestamps to bin data into five minute bins. Spatial bins are normalized between 0 and 255, using all times frames for normalization. Temporal bins are smoothed using a moving average of window size 3. 


## 1.CT image reconstruction  

We use **25 projections** to reconstruct the CT image. If the input is 50 projections, we downsample it to 25 projections. See the code in main.py at line 46.
```
python main.py --data_dir test/ --save_dir ./save --data_type gaussian
```
The data_type should be chosen from ['noisefree', 'gaussian', 'scattering'].

## 2.Defect detection  

```
python main.py --data_dir test/ --save_dir ./save
```
The input data is the reconstructed CT image, and the result is a mask as the ground truth.


# RCAN

This repository is implementation of the "Image Super-Resolution Using Very Deep Residual Channel Attention Networks".

<center><img src="./figs/fig2.png"></center>
<center><img src="./figs/fig3.png"></center>
<center><img src="./figs/fig4.png"></center>

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

For below results, we set the number of residual groups as 6, the number of RCAB as 12, the number of features as 64. <br />
In addition, we use a intermediate weights because training process need to take a looong time on my computer. 😭<br />

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x2</center></td>
        <td><center>RCAN x2</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_bicubic.png" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_RCAN.png" height="300"></center>
    	</td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.


### Test

Output results consist of restored images by the BICUBIC and the RCAN.


### 体会 
比赛最后生成的图片像素有点大，资源有限，自己掏钱买了一台1080ti电脑，吃了大半年土，奈何一块1080ti还是太low。
对一块显卡的人来说DBPN，RCAN层数必须要减少一下，否则测试会内存溢出，也试了下cvpr2019的SAN，也必须阉割一下。后面由于实习等原因放弃了，能学以致，这次比赛真的是不错的体验，一起加油吧

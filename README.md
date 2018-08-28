# Item2vec
Implement item2vec algorithm. I use the item name data from Ruten website.

## Usage

```
# start training
python calculate_similar_items.py --data data/data.csv

# start tensorboard to see the visualization
tensorboard --logdir=result/word/
# or
tensorboard --logdir=result/item/
```

## Visualization

### Word
<p align="center">
  <img src="images/item2vec-word.png" width="100%" />
</p>

### Item
<p align="center">
  <img src="images/item2vec-item.png" width="100%" />
</p>

## Result

* Example 1:

```
# Target
【NIXON Range Backpack 後背包 登山 機能款 C2390-158 1.0

# Similar
【NIXON Range Backpack 後背包 登山 機能款 C2390-000 1.0
NIXON FIELD BACKPACK 後背包 防潑水 大容量 登山 軍綠 C1577-10 0.974977708354
NIXON FIELD BACKPACK 後背包 防潑水 大容量 登山 C1577-306 現貨 0.958431276001
NIXON Grandview Backpack 後背包 旅行 防潑水 大容量 C2189-117 0.954813861429
NIXON Fountain Sling Pack 側背包 輕便 防潑水 多夾層 C1957-30 0.952343620246
NIXON Visitor Pack 後背包 旅行 防潑水 大容量 C2288-403 現貨 0.940520766744
日本直送附小票CHAMPION 雙LOGO後背包 BACKPACK SUPREME 雙肩書包 登山包旅行包現貨 0.917883300403
Dickies Texas德州3D立體刺繡字母簡潔美式風後背包STUSSY SUPREME登山雙肩書包 旅行包登山包 0.915754413273
日本實拍附小票CHAMPION 帆布雙肩包 BACKPACK書包復古 刺繡 後背包 登山包 古著MUJISTUSSY現貨 0.902719634174
```

* Example 2:
```
# Target
[【可到付】簡書N2a] 9787504553874 物流倉儲與配送管理實訓(簡體書) 1.0

# Similar
[【可到付】簡書N2a] 9787504554178 物流倉儲與配送管理(簡體書) 0.994554805494
[【可到付】簡書N2a] 9787562331780 倉儲與配送管理實務(簡體書) 0.988241956026
[【可到付】簡書N2a] 9787564033316 物流倉儲與配送(簡體書) 0.988132337738
[【可到付】簡書N2a] 9787562447955 配送管理(物流管理本科)(簡體書) 0.988091096129
[【可到付】簡書N2a] 9787561434970 物流倉儲管理(簡體書) 0.986730754724
[【可到付】簡書N2a] 9787122055170 物流倉儲管理(霍紅)(簡體書) 0.986730754724
[【可到付】簡書N2a] 9787508442877 現代物流倉儲與配送(簡體書) 0.986725138826
[【可到付】簡書N2a] 9787303086924 倉儲配送管理(簡體書) 0.986005867045
[【可到付】簡書N2a] 9787111300779 倉儲與配送管理 第2版(簡體書) 0.986005867045
```

* Example 3:
```
# Target
毛衣高領2017秋冬新款打底衫寬鬆韓版套頭學生軟妹網紅高領毛衣ulzzang女祕密盒子 1.0

# Similar
2017秋冬新款女裝寬鬆套頭衫韓版圓領純色內搭毛衣長袖打底針織衫祕密盒子 0.966702490158
韓國秋冬新款修身顯瘦長袖套頭針織衫小高領純色毛衣女打底衫 0.966699202363
秋冬季新款韓版高腰毛呢短褲休閒靴褲顯瘦外穿打底女闊腿褲短褲 0.966370230825
韓版秋冬季新款針織後開叉包臀半身裙子過膝中長款修身顯瘦【804】 0.961682251187
新款純色V領長袖t恤女韓版寬鬆大碼女裝白色簡約上衣純棉打底衫女祕密盒子 0.961150408888
秋冬2016新款外穿打底褲女裝修身顯瘦高腰啞光PU韓版小腳仿皮褲子 0.958171259676
新款純色圓領長袖t恤女純棉韓版修身上衣白色打底衫女士體恤衫潮祕密盒子 0.955428269964
男士秋季新款針織衫韓版修身2017潮流休閒薄款純色中長款毛衣開衫祕密盒子 0.954385510427
冬裝新款修身韓版條紋撞色高領長袖針織衫女學生百搭打底衫潮 0.950891601561
```

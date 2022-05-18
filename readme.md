加上每一个文件/文件夹的作用





the output is on the remote server and I don't move it down because it's big (it contains the trained model )

![image-20220517165026466](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220517165026466.png)



# about my test

`chmod u+x *.sh` active bash

`bash raw.sh`

to test the poison rate on the poison data

train on poison and test on poison

asr

![image-20220515152421274](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515152421274.png)



python test_pipeline_on_clean.py

get pipeline acc 

test on clean && raw model

![image-20220515153802586](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515153802586.png)

python test_pipeline_on_poisoned.py

get pipeline ASR



![image-20220517164933307](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220517164933307.png)

bash raw_on_clean.sh

to trai on poison and test on clean 

acc

![image-20220515160440048](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515160440048.png)



bash raw_clean.sh

train on clean test on clean



![image-20220515161207246](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515161207246.png)



bash raw_clean_poison.sh

train on clean test on poison 

![image-20220515164647981](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515164647981.png)

### word

#### pipeline method

python active_pipline.py

![image-20220514145111565](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514145111565.png)

./train_pipeline.sh 0 sst2 word 1 0.05 pipeline

![image-20220514002214956](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514002214956.png)

python align_predictions.py --dataset sst2 --type word --target 1 --rate 0.05 --defense pipeline

![image-20220514150025624](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514150025624.png)

./train_sanitized.sh 0 sst2 word 1 0.05 pipeline

![image-20220514002425120](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514002425120.png)

#### test human talent with label



python getseeds.py

result:

```
[(' ', 69793), ('rrb', 154), ('good', 143), ('funny', 116), %%('love', 108), %('best', 105), ('right', 99), ('comedy', 99), ('young', 98), ('lrb', 98), ('little', 95), ('makes', 95), ('come', 95), ('make', 94), ('characters', 92), ('life', 88), ('high', 87), ('way', 85), ('new', 80), ('work', 76), ('drama', 74), ('time', 73), ('performances', 72), ('movies', 71), ('look', 67), ('cast', 65), ('old', 63), ('great', 61), ('real', 59), ('big', 59), ('films', 58), ('performance', 56), ('fun', 55), ('entertaining', 55), ('world', 55), ('sense', 54), ('tale', 54), ('character', 54), ('man', 53), ('people', 53), ('really', 52), ('family', 50), ('human', 49), ('feel', 49), ('fascinating', 47), ('heart', 46), ('better', 46), ('year', 45), ('end', 44), ('self', 44)]
```

```
[(' ', 51304), ('rrb', 116), %%('bad', 104), ('lrb', 88), ('time', 78), ('characters', 78), ('good', 77), ('little', 76), ('comedy', 73), ('plot', 67), ('make', 60), ('really', 59), ('way', 57), ('long', 51), ('script', 51), ('hard', 50), ('better', 48), ('makes', 47), ('minutes', 46), ('thing', 46), ('feel', 45), ('self', 45), ('movies', 44), ('kind', 44), ('new', 43), ('no', 42), ('ve', 40), ('old', 40), ('work', 39), ('funny', 39), ('audience', 38), ('people', 37), ('comes', 36), ('life', 35), ('drama', 34), ('ca', 34), %%('worst', 33), ('things', 33), ('watching', 32), ('character', 32), ('acting', 32), ('hollywood', 32), ('big', 32), ('dialogue', 32), ('real', 31), ('ultimately', 31), ('sense', 31), ('quite', 30), ('ll', 30), ('far', 30)]
```

python prepare_two_seeds.py --dataset sst2 --type word --target 1 --rate 0.05

./train_two_seeds.sh 0 sst2 word 1 0.05

![image-20220514152043235](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514152043235.png)

python align_predictions.py --dataset sst2 --type word --target 1 --rate 0.05 --defense two_seeds

![image-20220514152256789](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514152256789.png)

./train_sanitized.sh 0 sst2 word 1 0.05 two_seeds

![image-20220517193945934](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220517193945934.png)

#### human without lable

python prepare_two_seedsraw.py --dataset sst2 --type word --target 1 --rate 0.05

![image-20220514153630082](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514153630082.png)



./train_brain_raw.sh 0 sst2 word 1 0.05

![image-20220514154104376](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514154104376.png)

python align_predictions.py --dataset sst2 --type word --target 1 --rate 0.05 --defense two_seeds_brain

![image-20220514154129354](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514154129354.png)

./train_sanitized.sh 0 sst2 word 1 0.05 two_seeds_brain

![image-20220514162932213](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220514162932213.png)



### sentence

#### human without label

python prepare_two_seedsraw.py --dataset sst2 --type sentence --target 1 --rate 0.05

![image-20220515134912599](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515134912599.png)

./train_brain_raw.sh 0 sst2 sentence 1 0.05

![image-20220515135316660](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515135316660.png)

python align_predictions.py --dataset sst2 --type sentence --target 1 --rate 0.05 --defense two_seeds_brain

![image-20220515135431245](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515135431245.png)

./train_sanitized.sh 0 sst2 sentence 1 0.05 two_seeds_brain

![image-20220515140431285](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515140431285.png)

#### human with label

python prepare_two_seeds.py --dataset sst2 --type sentence --target 1 --rate 0.05

![image-20220515141242411](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515141242411.png)

./train_two_seeds.sh 0 sst2 sentence 1 0.05

![image-20220515143927726](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515143927726.png)



python align_predictions.py --dataset sst2 --type sentence --target 1 --rate 0.05 --defense two_seeds



![image-20220515144004765](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515144004765.png)

./train_sanitized.sh 0 sst2 sentence 1 0.05 two_seeds

![image-20220515142243461](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515142243461.png)

#### python active_pipline.py

 ![image-20220515145226517](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515145226517.png)

./train_pipeline.sh 0 sst2 sentence 1 0.05 pipeline

![image-20220515150200637](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515150200637.png)

python align_predictions.py --dataset sst2 --type sentence --target 1 --rate 0.05 --defense pipeline

![image-20220515150217089](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515150217089.png)

./train_sanitized.sh 0 sst2 sentence 1 0.05 pipeline

![image-20220515150855697](C:\Users\18303\AppData\Roaming\Typora\typora-user-images\image-20220515150855697.png)





#### pipeline.py 可以测试小的样例 直接调用我训练好的模型
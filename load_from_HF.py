from crossformer.model.crossformer_model import CrossFormerModel
import pickle 

model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")

# model.save_pretrained(".")


# LOAD PRETRAINED WILL STORE IN YOUR CACHE SEE EXAMPLE BELOW
# IN CACHE YOU CAN FIND THE CHECKPOINT CONFIG AND OTHER FILES 
# YOU MIGHT NEED TO SCP THEM IN YOUR ROBOT SEE EXAMPLE BELOW
"""
(crossformer) ➜  crossformer git:(main) ✗ cd /home/mae/.cache/huggingface/hub/models--rail-berkeley--crossformer/snapshots/c7dea2691aed3656537c5126a0a77df84a28abd7        git:(main|✚1-2…7 
(crossformer) ➜  c7dea2691aed3656537c5126a0a77df84a28abd7 ls -lisa
total 16
72484586 4 drwxrwxr-x 3 mae mae 4096 Aug 24 16:21 .
72484585 4 drwxrwxr-x 3 mae mae 4096 Aug 24 16:21 ..
72484590 4 drwxrwxr-x 3 mae mae 4096 Aug 24 16:21 300000
72484600 0 lrwxrwxrwx 1 mae mae   52 Aug 24 16:21 config.json -> ../../blobs/120ad7b218b1085344c1fcaeff15c0d5d4e9c81a
72484605 0 lrwxrwxrwx 1 mae mae   52 Aug 24 16:21 dataset_statistics.json -> ../../blobs/db54cd39a1a558a4f8d2df844025a2d56a746876
72484612 4 lrwxrwxrwx 1 mae mae   76 Aug 24 16:21 example_batch.msgpack -> ../../blobs/f754caf846b6ad83f66a2076a2a1165e785dd0e67ab9f495bcf6743316d2fd5a
72484603 0 lrwxrwxrwx 1 mae mae   52 Aug 24 16:21 .gitattributes -> ../../blobs/0a7d438d1a5b51d839a2af7fe8a1e60ee29afca7
72484604 0 lrwxrwxrwx 1 mae mae   52 Aug 24 16:21 README.md -> ../../blobs/bad3429819fb606abba5009468ee7836dd981b2d
(crossformer) ➜  c7dea2691aed3656537c5126a0a77df84a28abd7 scp config.json xavier@192.168.1.23:/ssd/SafeGNM/crossformer/assets/
xavier@192.168.1.23's password: 
config.json                                                                                                                                                100%   63KB   1.7MB/s   00:00    
(crossformer) ➜  c7dea2691aed3656537c5126a0a77df84a28abd7 scp dataset_statistics.json xavier@192.168.1.23:/ssd/SafeGNM/crossformer/assets/
xavier@192.168.1.23's password: 
dataset_statistics.json                                                                                                                                    100%   84KB   1.7MB/s   00:00    
(crossformer) ➜  c7dea2691aed3656537c5126a0a77df84a28abd7 scp example_batch.msgpack xavier@192.168.1.23:/ssd/SafeGNM/crossformer/assets/  
xavier@192.168.1.23's password: 
example_batch.msgpack                                                                                                                                      100% 4450KB   5.0MB/s   00:00    
(crossformer) ➜  c7dea2691aed3656537c5126a0a77df84a28abd7 





"""
from crossformer.model.crossformer_model import CrossFormerModel
import pickle 

model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")

# model.save_pretrained(".")


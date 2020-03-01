# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer


@registry.register_model("entitygrid_model")
class EntityGrid(Pythia):
    def __init__(self, config):
        super().__init__(config)

        img_in_channels = 2048
        ocr_in_channels = 300
        joint_in_channels = 512
        joint_middle_channels = 768

        out_channels = 256
        joint_out_channels = 1024

        kernel_size = 3
        stride = 1
        padding = 0 

        pool_stride = 2
        pool_kernel = 2
        #TODO: use nn.Sequential

        act_f = nn.ReLU()

        self.img_net = nn.Sequential(
            nn.Conv2d(img_in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            act_f
        )

        self.chargrid_net = nn.Sequential(
            nn.Conv2d(ocr_in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            act_f
        )

        self.entitygrid_net = nn.Sequential(
            nn.Conv2d(joint_in_channels, joint_in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(joint_in_channels),
            act_f,
            nn.MaxPool2d(pool_kernel, stride=pool_stride),
            nn.Conv2d(joint_in_channels, joint_middle_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(joint_middle_channels),
            act_f,
            nn.MaxPool2d(pool_kernel, stride=pool_stride),
            nn.Conv2d(joint_middle_channels, joint_out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(joint_out_channels),
            nn.MaxPool2d(pool_kernel, stride=pool_stride)
        )


      """   #Image CNN
        self.conv1 = nn.Conv2d(img_in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        #OCR CNN
        self.conv2 = nn.Conv2d(ocr_in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        #Joint CNNs
        self.conv3 = nn.Conv2d(joint_in_channels, joint_in_channels, kernel_size, stride, padding)
        self.batchnorm3 = nn.BatchNorm2d(joint_in_channels)
        self.max_pool2d3 = nn.MaxPool2d(pool_kernel, stride=pool_stride)

        self.conv4 = nn.Conv2d(joint_in_channels, joint_middle_channels, kernel_size, stride, padding)
        self.batchnorm4 = nn.BatchNorm2d(joint_middle_channels)
        self.max_pool2d4 = nn.MaxPool2d(pool_kernel, stride=pool_stride)

        self.conv5 = nn.Conv2d(joint_middle_channels, joint_out_channels, kernel_size, stride, padding)
        self.batchnorm5 = nn.BatchNorm2d(joint_out_channels)
        self.max_pool2d5 = nn.MaxPool2d(pool_kernel, stride=pool_stride) """

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
        ]

        return params

    def _get_classifier_input_dim(self):
        # Now, the classifier's input will be cat of image and context based
        # features
        #return 2 * super()._get_classifier_input_dim()
        return 2 * super()._get_classifier_input_dim()
        

    def forward(self, sample_list):
        #sample list contains now also 
        sample_list.text = self.word_embedding(sample_list.text)
        #(batch_size x 4096)
        text_embedding_total = self.process_text_embedding(sample_list)

        #img_feat_canvas (batch_size x 2048 x 128 x 128)
        #ocr_feat_canvas (batch_size x 300  x 128 x 128)

        #sparse tensors are experimential

        #CNN
        #img_emb = F.relu(self.batchnorm1(self.conv1(sample_list.img_feat_canvas)))
        #ocr_emb = F.relu(self.batchnorm2(self.conv2(sample_list.ocr_feat_canvas)))
        img_emb = self.img_net(sample_list.img_feat_canvas)
        ocr_emb = self.chargrid_net(sample_list.ocr_feat_canvas)
        #back prop has to be applied, no deletion possible
        #del sample_list["img_feat_canvas"]
        #del sample_list["ocr_feat_canvas"]

        #torch.cuda.empty_cache()

        

        joint_emb = torch.cat([img_emb,ocr_emb],1)
        joint_emb = self.entitygrid_net(joint_emb)

        """ joint_emb = self.max_pool2d3(
            F.relu(
                self.batchnorm3(
                    self.conv3(joint_emb))))

        joint_emb = self.max_pool2d4(
            F.relu(
                self.batchnorm4(
                    self.conv4(joint_emb))))

        joint_emb = self.max_pool2d5(
            F.relu(
                self.batchnorm5(
                    self.conv5(joint_emb))))
 """
        #batch_size x channel x height x width


        #Flatten feature vector
        #TODO: get official batch size
        batch_size = joint_emb.shape[0]
        feature_vector_size = joint_emb.shape[1]
        #print(joint_emb.shape)
        joint_emb = joint_emb.view((batch_size,feature_vector_size,-1))
        joint_emb = joint_emb.permute(0,2,1)

        #Add feature embeddings to sample_list
        setattr(sample_list,"image_feature_0",joint_emb)
        
        #TODO: use max features from config
        #max features 
        feature_dim = torch.zeros((batch_size),dtype=torch.int64)
        feature_dim[:] = 196
        feature_info = MaxFeatureClass(feature_dim)
        
        setattr(sample_list,"image_info_0",feature_info)

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        context_embedding_total, _ = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image","text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores}

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

class MaxFeatureClass:
    def __init__(self,max_features):
        self.max_features = max_features
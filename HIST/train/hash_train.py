from model.hash_model import DCMHT as DCMHT
import os
from nasa.nasaloss import *
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from dataset.dataloader import dataloader
import numpy as np
# from SupConLoss import *
from SelectiveSupConLoss import *
import torch.nn.functional as F
from histloss.hist import CDs2Hg,HGNN
from histloss.contextual import margin_contrastive
from labelnet import LabelNet
import torchvision.transforms as transforms
from AdaQuadruplet import AdaQuadrupletLoss
from test import MultiPosPairLoss
from DFML import  MarginLoss

class Trainer(TrainBase):

    def __init__(self,
                 rank=0):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        # bit = args.output_dim
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        linear = False
        if self.args.hash_layer == "linear":
            linear = True

        self.logger.info("ViT+GPT!")
        HashModel = DCMHT
        self.ClassLen = 0
        if self.args.dataset == 'nuswide':
            self.ClassLen = 21
        elif self.args.dataset == 'flickr25k':
            self.ClassLen = 24
        elif self.args.dataset == 'coco':
            self.ClassLen = 80
        else:
            self.ClassLen = 291

        self.lossMS_I2T_list = np.array([])
        self.lossMS_I2I_list = np.array([])
        self.lossMS_T2T_list = np.array([])

        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                               writer=self.writer, logger=self.logger, is_train=self.args.is_train, linear=linear).to(
            self.rank)
        self.cdg2_i = CDs2Hg(nb_classes=self.ClassLen,sz_embed=self.args.output_dim,device=self.rank).to(self.rank)
        self.cdg2_t = CDs2Hg(nb_classes=self.ClassLen,sz_embed=self.args.output_dim,device=self.rank).to(self.rank)
        self.hgnn_i = HGNN(nb_classes=self.ClassLen,sz_embed=self.args.output_dim,hidden=1024,device=self.rank).to(self.rank)
        self.hgnn_t = HGNN(nb_classes=self.ClassLen,sz_embed=self.args.output_dim,hidden=1024,device=self.rank).to(self.rank)
        self.project_i = ProjectLayer(512).to(self.rank)
        self.project_t = ProjectLayer(512).to(self.rank)
        self.triplet = DTSHLoss()
        self.nasa_loss = NASA_loss(device=self.rank)
        self.criterion = nn.CrossEntropyLoss()
        self.ADQ = AdaQuadrupletLoss().to(0)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        self.dpsh = DPSHLoss(num_train=self.train_labels.shape[0],nclass=self.ClassLen,bit=self.args.output_dim,device=self.rank)
        self.model.float()
        # self.supervision.float()

        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.triplet.parameters(), 'lr': self.args.lr},
            {'params': self.cdg2_i.parameters(), 'lr': self.args.lr},
            {'params': self.cdg2_t.parameters(), 'lr': self.args.lr},
            {'params': self.hgnn_i.parameters(), 'lr': self.args.lr},
            {'params': self.hgnn_t.parameters(), 'lr': self.args.lr},
            {'params': self.project_i.parameters(), 'lr': self.args.lr},
            {'params': self.project_t.parameters(), 'lr': self.args.lr},
        ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        # print(self.model)

    def _init_dataset(self):
        self.aug = torch.nn.Sequential(
            transforms.RandomHorizontalFlip()
        )
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_dataset = train_data
        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False
        )

    # YOCO in batch-level, if h/w size is odd, you might need to manually change the YOCO part
    # see YOCO in Contrastive/moco/loader.py for YOCO in PIL image level

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        i_lossl = 0
        t_lossl = 0
        it_lossl = 0
        times = 0

        for image, text, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image.float()
            image = image.to(self.rank, non_blocking=False)
            text = text.to(self.rank, non_blocking=False)
            labels = label.float().to(self.rank)

            self.optimizer.zero_grad()
            hash_img,st_i, hash_text,st_t = self.model(image, text)

            dist_loss1 , Hi = self.cdg2_i(hash_img,labels)
            dist_loss2 , Ht = self.cdg2_t(hash_text,labels)
            #
            Hi , Ht = Hi.to(self.rank) , Ht.to(self.rank)
            out_i , out_t = self.hgnn_i(hash_img,Hi),self.hgnn_t(hash_text,Ht)
            cls_loss_i , cls_loss_t = self.criterion(out_i,labels),self.criterion(out_t,labels)

            # i, n = self.ADQ(hash_img, labels, hash_img)
            # t,m  = self.ADQ(hash_text, labels, hash_text)
            # it, nm = self.ADQ(hash_img, labels, hash_text)

            nasa_i , nasa_t = self.nasa_loss(st_i,self.project_i(st_i)),self.nasa_loss(st_t,self.project_t(st_t))
            i_loss ,t_loss = self.triplet(hash_img,labels), self.triplet(hash_text,labels)
            it_loss = self.triplet(hash_img,labels,hash_text)

            tot_hier = cls_loss_t + cls_loss_i + (it_loss + i_loss + t_loss) + (nasa_i + nasa_t)
            # all_loss +=  tot_hier
            # all_loss += it_loss + i_loss + t_loss + nasa_i + nasa_t
            all_loss += cls_loss_t + cls_loss_i + (it_loss + i_loss + t_loss) + (nasa_i + nasa_t)
            tot_hier.backward()

            self.optimizer.step()

        # -   >>>>>>> FINISHED >>>>>> Best epoch, I-T: 102, mAP: 0.8243458271026611, T-I: 112, mAP: 0.8152998089790344 32
        # 03/27/2024 12:59:47 - INFO -   >>>>>> MAX MAP(i->t): 0.79723060131073, MAX MAP(t->i): 0.7927498817443848 16
        #  >>>>>>> FINISHED >>>>>> Best epoch, I-T: 128, mAP: 0.842654287815094, T-I: 91, mAP: 0.8177953958511353 64
        #    >>>>>>> FINISHED >>>>>> Best epoch, I-T: 144, mAP: 0.8431671857833862, T-I: 52, mAP: 0.8154277801513672
        torch.cuda.empty_cache()
        self.lossMS_I2I_list = np.append(self.lossMS_I2I_list, i_lossl / len(self.train_loader))
        self.lossMS_T2T_list = np.append(self.lossMS_T2T_list, t_lossl / len(self.train_loader))
        self.lossMS_I2T_list = np.append(self.lossMS_I2T_list, it_lossl / len(self.train_loader))
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")
        self.logger.info(
            f">>>>>> MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")
    def train(self):
        self.logger.info("Start train.")

        self.scalar = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            # if epoch > 100:
            self.valid(epoch)
            self.save_model(epoch)
        np.savetxt("TrainMS_M_III", self.lossMS_I2I_list, delimiter=',')
        np.savetxt("TrainMS_M_TTT", self.lossMS_T2T_list, delimiter=',')
        np.savetxt("TrainMS_M_IT", self.lossMS_I2T_list, delimiter=',')
        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def test(self, model='i2t'):
        self.logger.info("test")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = super().get_code(self.retrieval_loader, self.args.retrieval_num,)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = super().get_code(self.retrieval_loader, self.args.retrieval_num, )
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=mAPi2t)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i", map=mAPt2i)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=0):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir,
                                  str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + f'{map:.4f}_.mat'),
                     result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")



import os
from tqdm import tqdm
import torch

def train(model, 
          train_loader, 
          val_loader, 
          optim, 
          metrics, 
          seg_loss, 
          dep_loss, 
          scheduler,
          save_name, 
          epochs = 100, 
          path_for_models =os.path.split(os.path.dirname(os.getcwd()))[0]+'/weights', 
          device=torch.device('cuda:0'), 
          train_mode = 'both'):
    try:
        os.mkdir(path_for_models)
    except:
        path_for_models=path_for_models
        
    stop_count = 0
    stop_iter = 10
    best_iou = -1
    min_loss = 10

    print("="*13+train_mode+"="*13)
    
    for epoch in range(epochs):
      total_seg_train_loss = 0.0
      total_dep_train_loss = 0.0
      val_total_seg_loss = 0.0
      val_total_dep_loss = 0.0
      
      with tqdm(train_loader,desc='Train',unit='batch') as tepoch:
        t_ious=[]
        for i, (X_train,y_train, d_label) in enumerate(tepoch):
          X_train, y_train, d_label=X_train.to(device),y_train.to(device,dtype=torch.int64), d_label.to(device)
          
          model.train()

          y_pred, d_pred =model(X_train) 
          # print(d_pred.shape) : torch.Size([16, 1, 256, 256])
          s_loss = seg_loss(y_pred, y_train.float()) #
          d_loss = dep_loss(d_pred, d_label.float())

          if train_mode == 'dep' : loss = d_loss
          elif train_mode == 'seg' : loss = s_loss
          elif train_mode == 'both': loss = d_loss + s_loss
          elif train_mode == 'both85' : loss = 0.15 * d_loss + 0.85* s_loss
          
          tiou = metrics(y_pred,y_train)
          t_ious.append(tiou)

          optim.zero_grad()
          loss.backward()
          optim.step()

          total_seg_train_loss += s_loss.sum().item()
          total_dep_train_loss += d_loss.sum().item()
          tepoch.set_postfix(seg_loss = total_seg_train_loss/(i+1), dep_loss = total_dep_train_loss/(i+1))
          
      with tqdm(val_loader, desc = "Valid", unit= 'batch') as tepoch:
        ious=[]
        val_s_loss = []
        val_d_loss = []
        with torch.no_grad():
          for b,(X_test,y_test, d_label) in enumerate(tepoch):
            X_test, y_test, d_label=X_test.to(device),y_test.to(device), d_label.to(device)
          
            model.eval()

            y_val, d_pred_val = model(X_test)
            
            vs_loss = seg_loss(y_val,y_test)
            vd_loss = dep_loss(d_pred_val, d_label)

            val_s_loss.append(vs_loss)
            val_d_loss.append(vd_loss)

            iou_= metrics(y_val,y_test)
            ious.append(iou_)
            
            val_total_seg_loss += vs_loss.sum().item()
            val_total_dep_loss += vd_loss.sum().item()
            
            tepoch.set_postfix(seg_loss = val_total_seg_loss/(b+1), dep_loss = val_total_dep_loss/(b+1),iou = torch.tensor(ious).mean())

          ious=torch.tensor(ious)
          t_ious = torch.tensor(t_ious)

          val_losses = torch.tensor(val_s_loss)
          scheduler.step(val_losses.mean())
          
          if ious.mean() > best_iou:
            best_iou=ious.mean()
            torch.save(model.state_dict(),f"{path_for_models}/{save_name}.pt")
            stop_count = 0
          else: 
            stop_count += 1


      print(f"epoch : {epoch:2} train_seg_loss: {s_loss:5.4} , val_seg_loss : {val_losses.mean():.4f}, train_iou: {t_ious.mean():.4f}, val_iou: {ious.mean():.4f}")
      
      print('='*30)
      if stop_count >= stop_iter:
        break
    

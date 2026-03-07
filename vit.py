import torch
import random
import numpy as np
from torch import autocast

import torch.nn as nn

from torchvision import transforms

from torchvision import datasets  # get dataset
from torch.utils.data import DataLoader  # get dataloader

import math

import time
from tqdm import tqdm

import os

from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'imagenet')
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.embed(x)  # [B,embed_dim,H/P,W/P]
        x = x.flatten(2).transpose(1, 2)  # [B,num_patches(H/P*W/P),embed_dim]
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop_rate=0., drop_rate=0., qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1,
                                                              4)  # [B,N,3*D]->[B,N,3,H,D//H]->[3,B,H,N,D//H]. wrongly used D/H instead of // and got float
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,D//H]
    
        attn = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          dropout_p=self.attn_drop.p if self.training else 0.0)  # [B,H,N,D//H]
        attn = attn.transpose(1, 2)  # [B,N,H,D//H]
        x = attn.reshape(B, N, D)  # [B,N,D]
        x = self.proj(x)  # [B,N,D] 
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, activation=nn.GELU, drop_rate=0.):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop_out = nn.Dropout(drop_rate)  # forgot

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)  # forgot

        return x


class DropPath(nn.Module):
    def __init__(self, drop_path_rate=0.):
        super().__init__()
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        if self.training and self.drop_path_rate > 0.:
            if torch.rand(1).item() < self.drop_path_rate:
                return x * 0.
            else:
                return x / (1 - self.drop_path_rate)
        else:
            return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, norm=nn.LayerNorm, attn_drop_rate=0., drop_rate=0.,
                 qkv_bias=True, drop_path_rate=0.,
                 activation=nn.GELU):  # embed_dim(of attention) = in_dim(of MLP), created both redundantly
        super().__init__()
        self.norm1 = norm(dim)  # dim forgot
        self.attn = Attention(dim, num_heads, attn_drop_rate, drop_rate, qkv_bias)
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = norm(dim)
        self.MLP = MLP(dim, 4 * dim, activation=activation, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.MLP(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, drop_path_rate, depth, num_heads,
                 attn_drop_rate, drop_rate, qkv_bias,
                 num_classes):  # except attn_drop_rate exclusive for attention, all other dropout(proj,mpl) use drop_rate!
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, nn.LayerNorm, attn_drop_rate, drop_rate, qkv_bias, dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_params)  # func name is enough?no param pass in?

    def _init_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,
                                  0.0)  # 1.bias of Linear could be None! forgot; 2.should use constant_, used trunc_normal_ mistakenly

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x):
        x = self.patch_embed(x)

        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed  # 1.'x+' missed; 2.put x in pos_embed mistakenly(pos_embed is not callable)
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def create_vit():
    model = ViT(image_size=224,
                patch_size=16,
                embed_dim=768,
                drop_path_rate=0.1,
                depth=12,
                num_heads=12,
                attn_drop_rate=0.0,
                drop_rate=0.0,
                qkv_bias=True,
                num_classes=1000)

    return model


class Config:
    train_dir = os.environ.get('IMAGENET_TRAIN_DIR', os.path.join(DEFAULT_DATA_ROOT, 'train'))
    val_dir = os.environ.get('IMAGENET_VAL_DIR', os.path.join(DEFAULT_DATA_ROOT, 'val'))

    batch_size = 256
    grad_accumulate_steps = 4

    num_workers = 8
    pin_memory = True

    mix_prob = 0.8
    cutmix_prob = 0.8
    mixup_alpha = 0.8
    cutmix_alpha = 1.0

    lr = 5e-4
    betas = (0.9, 0.999)  # used (0.99,0.999) mistakenly
    eps = 1e-8
    weight_decay = 0.05

    total_epochs = 300
    warmup_epochs = 20
    min_lr = 1e-6

    label_smoothing_rate = 0.1

    checkpoint_path = os.environ.get(
        'VIT_CHECKPOINT_PATH',
        os.path.join(DEFAULT_OUTPUT_ROOT, 'checkpoints', 'vit_imagenet_best.pth')
    )
    log_dir = os.environ.get(
        'VIT_TENSORBOARD_DIR',
        os.path.join(DEFAULT_OUTPUT_ROOT, 'runs', 'vit_imagenet')
    )
    patience = 30
    min_delta = 0.01

    print_freq = 10


class ImageNetAugmentation:
    def __init__(self, is_training):
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandAugment(2, 9),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(0.25)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])

    def __call__(self, x):
        return self.transform(x)  # x missed


def get_data_loaders(config):
    train_aug = ImageNetAugmentation(True)
    val_aug = ImageNetAugmentation(False)

    train_dataset = datasets.ImageFolder(root=config.train_dir, transform=train_aug)
    val_dataset = datasets.ImageFolder(root=config.val_dir, transform=val_aug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=True
    )

    return train_loader, val_loader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def accuracy(output, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)  # used size[0] mistakenly

    _, pred = output.topk(maxk, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracy_k = correct_k.mul_(100.0 / batch_size)
        res.append(accuracy_k)
    return res


def mixup_data(input, target, alpha):
    lam = np.random.beta(alpha, alpha)
    B = input.shape[0]
    index = torch.randperm(B, device=input.device)
    mixed_input = lam * input + (1 - lam) * input[index, :]
    mixed_target = target[index]

    return mixed_input, target, mixed_target, lam


def cutmix_data(input, target, alpha):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = input.shape
    index = torch.randperm(B, device=input.device)  # used np instead of torch mistakenly

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_ratio))
    cut_h = max(1, int(H * cut_ratio))

    cx = np.random.randint(cut_w // 2, W - cut_w // 2 + 1)  # used torch.randint mistakenly
    cy = np.random.randint(cut_h // 2, H - cut_h // 2 + 1)  # write W - cut_h // 2 + 1 mistakenly

    bbx1 = cx - cut_w // 2
    bbx2 = cx + cut_w // 2
    bby1 = cy - cut_h // 2
    bby2 = cy + cut_h // 2

    input[:, :, bbx1:bbx2, bby1:bby2] = input[index, :, bbx1:bbx2,
                                        bby1:bby2]  # assigned to mixed_input mistakenly(patch,not the whole image)
    mixed_input = input
    mixed_target = target[index]

    lam = 1 - (cut_w * cut_h) / (W * H)

    return mixed_input, target, mixed_target, lam


def mix_criterion(criterion, output, target, mixed_target, lam):
    return lam * criterion(output, target) + (1 - lam) * criterion(output, mixed_target)


def mix_accuracy(output, targets, mixed_targets, lam, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)  # used size[0] mistakenly

    _, pred = output.topk(maxk, 1, True)
    pred = pred.t()
    correct_a = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_b = pred.eq(mixed_targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = lam*correct_a[:k].reshape(-1).float().sum(0, keepdim=True)+(1-lam)*correct_b[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracy_k = correct_k.mul_(100.0 / batch_size)
        res.append(accuracy_k)
    return res


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smooth_rate):
        super().__init__()
        self.smooth_rate = smooth_rate
        self.confidence = 1.0 - smooth_rate

    def forward(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        nll = -log_prob[range(target.shape[0]), target]
        smooth = -log_prob.mean(dim=-1)
        loss = nll * self.confidence + smooth * self.smooth_rate

        return loss.mean()


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.min_lr = min_lr

        # self.init_lr=self._calculate_lr(0)
        # for param_group in optimizer.param_groups:
        #    param_group['lr']=self.init_lr

    def step(self):
        self.current_epoch += 1  # moved before new lr assignment(should update to new lr first)

        lr = self._calculate_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _calculate_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (epoch + 1) / self.warmup_epochs * self.base_lr  # forgot +1#epoch write as self.current_epoch mistakenly
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(progress * math.pi))

        return lr

    def state_dict(self):
        return {
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr
        }

    def load_state_dict(self, state_dict):
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_epoch(config, model, train_loader, device, criterion, scaler, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    optimizer.zero_grad()

    end = time.time()

    pbar = tqdm(train_loader, desc='Train')
    for batch_idx, (inputs, targets) in enumerate(pbar):  # enumerate forgot
        data_time.update(time.time() - end)
        end = time.time()

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mix = torch.rand(1).item() < config.mix_prob

        if use_mix:
            use_cutmix = torch.rand(1).item() < config.cutmix_prob
            if use_cutmix:
                # should assign to inputs. If assign to other names, they are not on GPU.
                inputs, targets, mixed_targets, lam = cutmix_data(inputs, targets, config.cutmix_alpha)
            else:
                inputs, targets, mixed_targets, lam = mixup_data(inputs, targets, config.mixup_alpha)

        with autocast('cuda'):
            outputs = model(inputs)
            if use_mix:
                loss = mix_criterion(criterion, outputs, targets, mixed_targets, lam)
            else:
                loss = criterion(outputs, targets)

        loss = loss / config.grad_accumulate_steps#forgot to /config.grad_accumulate_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.grad_accumulate_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        if use_mix:
            acc1, acc5 = mix_accuracy(outputs, targets, mixed_targets,lam,(1, 5))
        else:
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
        top1.update(acc1.item(), targets.shape[0])
        top5.update(acc5.item(), targets.shape[0])

        losses.update(loss * config.grad_accumulate_steps, targets.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.print_freq == 0:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1.avg:.2f}%',
                'Top5': f'{top5.avg:.2f}%',
                'Batch Time': f'{batch_time.avg:.3f}s',
                'Data Time': f'{data_time.avg:.3f}s'
            })

    if len(train_loader) % config.grad_accumulate_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

    return losses.avg, top1.avg, top5.avg


def val_epoch(model, val_loader, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, desc='Validate')

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            top1.update(acc1.item(), targets.shape[0])
            top5.update(acc5.item(), targets.shape[0])

            losses.update(loss, targets.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1.avg:.2f}%',
                'Top5': f'{top5.avg:.2f}%',
                'Batch Time': f'{batch_time.avg:.3f}s'
            })

        return losses.avg, top1.avg, top5.avg


def resume_training(checkpoint_path, device, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint['best_acc1']

        print(f'Resuming training from epoch {start_epoch + 1} with best top1 accuracy {best_acc1:.2f}%')
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
        if optimizer.param_groups[0]['lr'] != scheduler._calculate_lr(scheduler.current_epoch):
            print(f'Asynchronization discovered:')
            print(f'Optimizer Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
            print(f'Scheduler Learning Rate: {scheduler._calculate_lr(scheduler.current_epoch):.8f}')

        return start_epoch, best_acc1
    else:
        init_lr = scheduler._calculate_lr(0)
        for param_group in optimizer.param_groups:  # added for lr synchronization of epoch 0.
            param_group['lr'] = init_lr
        print('No checkpoint found, starting from scratch!')
        return 0, 0.0


def main():
    config = Config()
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():  # used device=='cuda' mistakenly (device is not a str!)
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(
            f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')  # added () after total_memory mistakenly (not callable)

    train_loader, val_loader = get_data_loaders(config)
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Classes: {len(train_loader.dataset.classes)}')  # loader doesn't have 'classes', dataset has

    model = create_vit().to(device)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    if config.label_smoothing_rate > 0:
        criterion = LabelSmoothingLoss(config.label_smoothing_rate)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )

    scheduler = WarmupCosineScheduler(optimizer, config.total_epochs, config.warmup_epochs, config.min_lr)

    start_epoch, best_acc1 = resume_training(config.checkpoint_path, device, model, optimizer, scheduler)

    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    writer = SummaryWriter(config.log_dir)

    scaler = GradScaler()

    patience_counter = 0

    print('Starting training...')
    print('=' * 80)

    for epoch in range(start_epoch, config.total_epochs):
        print(f'Epoch {epoch + 1}/{config.total_epochs}')  # +1 forgot
        print('-' * 60)

        current_lr = scheduler.get_current_lr()

        train_loss, train_top1, train_top5 = train_epoch(config, model, train_loader, device, criterion, scaler,
                                                         optimizer)
        val_loss, val_top1, val_top5 = val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        new_lr = scheduler.get_current_lr()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validate', val_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top1', train_top1, epoch)
        writer.add_scalar('Accuracy/Validate_Top1', val_top1, epoch)
        writer.add_scalar('Accuracy/Train_Top5', train_top5, epoch)
        writer.add_scalar('Accuracy/Validate_Top5', val_top5, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f'Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.2f}% | Train Top5: {train_top5:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.2f}% | Val Top5: {val_top5:.2f}%')
        print(f'Learning Rate: {current_lr:.8f} -> {new_lr:.8f}')

        if val_top1 >= best_acc1 + config.min_delta:  # write min_delta as min_lr mistakenly
            best_acc1 = val_top1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'config': config  # forgot
            }
            torch.save(checkpoint, config.checkpoint_path)

            print(f'NEW BEST MODEL saved with best top1 accuracy: {best_acc1:.2f}%')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{config.patience}')

        if patience_counter >= config.patience:
            print(f'Early stopping after {epoch + 1} epochs of training!')
            break

        print('=' * 80)

    print('Training completed!')
    print(f'Best top1 accuracy: {best_acc1:.2f}%')

    try:
        checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss, val_top1, val_top5 = val_epoch(model, val_loader, criterion, device)
        print(f'Final evaluation:')
        print(f'Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.2f}% | Val Top5: {val_top5:.2f}%')
        print(f"Best epoch: {checkpoint['epoch'] + 1}")

    except Exception as e:
        print(f'Exception occurred during final checkpoint loading: {e}')

    writer.close()
    print('Training completed successfully!')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Training interrupted by user')
    except Exception as e:
        print(f'Exception: {e}')
        raise

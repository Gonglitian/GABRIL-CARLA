# Rclone 配置与使用总结

本文档总结了从在无 `sudo` 权限的服务器上安装 rclone，到配置 Google Drive、挂载使用、文件操作与注意事项，以及 GUI 可视化界面的全过程。并记录了过程中遇到的常见问题与解答。

---

## 1. 安装 Rclone（无 sudo 权限）

### 下载与解压
```bash
cd ~
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-*-linux-amd64.zip
cd rclone-*-linux-amd64
mkdir -p ~/bin
cp rclone ~/bin/
chmod +x ~/bin/rclone
```

### 配置 PATH
编辑 `~/.bashrc`，加入：

```bash
export PATH="$HOME/bin:$PATH"
```

然后让其生效：

```bash
source ~/.bashrc
```

### 验证

```bash
which rclone
rclone version
```

## 2. 配置 Google Drive Remote

### 初始化配置
```bash
rclone config
```

1. 选择 `n` → 新建 remote
2. Storage 选择 `Google Drive`
3. **client_id / client_secret**
   - 临时使用：直接回车留空，使用默认 ID（限额较低）
   - 长期使用：在 Google Cloud Console 申请专属 ID 与 Secret
4. **Scopes**
   - `1 (drive)`：完全访问（推荐）
   - `2 (drive.readonly)`：只读访问
   - 其他选项不常用
5. **service_account_file**：直接回车留空（除非要用 Google Cloud Service Account）
6. **Advanced config**：输入 `n`
7. **Auto config？**
   - 若服务器无浏览器 → 输入 `n`
   - 在本地有浏览器机器运行提示的：
     ```bash
     rclone authorize "drive" "<json string>"
     ```
   - 浏览器授权后获得 token，粘贴回服务器。

### 配置文件位置
最终配置文件保存在：
```
~/.config/rclone/rclone.conf
```

可以通过以下命令查看 remote：
```bash
rclone listremotes
```

## 3. 挂载 Google Drive

### 创建挂载点
```bash
mkdir -p /scr/litian/gdrive
```

### 挂载命令
```bash
rclone mount litian_LIRA: /scr/litian/gdrive --daemon --vfs-cache-mode full
```

**参数说明：**
- `--daemon`：后台运行，不依赖 SSH 会话
- `--vfs-cache-mode full`：启用完整缓存，写入文件后后台上传到远端
- 可选：`--drive-use-trash` 删除时进入垃圾桶而不是直接永久删除
- 可选：`--allow-other` 允许其他用户访问

### 卸载
```bash
fusermount -u /scr/litian/gdrive
# 或
fusermount3 -u /scr/litian/gdrive
```

## 4. 挂载后的使用说明

### 文件与目录操作
- **创建目录：**
  ```bash
  mkdir /scr/litian/gdrive/my_checkpoints
  ```

- **保存 PyTorch 模型：**
  ```python
  torch.save(model.state_dict(), "/scr/litian/gdrive/my_checkpoints/model_epoch_1.ckpt")
  ```

### 缓存机制
- 写入先进入本地缓存，再后台异步上传
- 删除会同步到远端 → 默认永久删除，除非加 `--drive-use-trash`

### 删除文件的注意事项
- **直接 `rm`：** 会删除远端文件，且可能导致大文件删除时卡顿
- **更安全的做法：**
  ```bash
  rclone delete litian_LIRA:path/to/dir --recursive --progress
  rclone purge litian_LIRA:path/to/dir --progress
  ```

## 5. 使用 rclone 命令上传/移动文件

### 上传文件（保留本地）
```bash
rclone copy ~/models/checkpoint.ckpt litian_LIRA:Backups --progress
```

### 移动文件（上传后删除本地）
```bash
rclone move ~/models/checkpoint.ckpt litian_LIRA:Backups --progress
```

### 常用参数
- `--progress` 显示进度条
- `--transfers=N` 并行上传数
- `--dry-run` 模拟执行，不实际操作

## 6. 常见问题与解答

**Q: 为什么 remote 名称后面要加冒号？**
A: `litian_LIRA:` 里的冒号是 rclone 的语法，用来区分远程和本地路径。没有冒号会被认为是本地路径。

**Q: 在挂载目录下删除文件会怎样？**
A: 默认会同时删除远端文件。除非加 `--drive-use-trash`，否则不会进垃圾桶。

**Q: 我删除文件后 SSH 卡死了？**
A: 大量删除可能导致挂载层阻塞，尤其在前台运行时容易卡死。解决办法：
- 用 `--daemon` 运行挂载
- 用 `rclone delete`/`rclone purge` 删除文件

**Q: 文件是不是自动下载到本地了？**
A: 挂载目录中看到的文件只是"虚拟映射"。只有访问时才会按需下载对应部分。

## 7. Rclone 可视化界面

### 官方 Web GUI
```bash
rclone rcd --rc-web-gui
```

- 默认运行在 `http://localhost:5572`
- 可加 `--rc-user` 和 `--rc-pass` 保护访问
- 可远程访问：
  ```bash
  rclone rcd --rc-web-gui --rc-addr :5572 --rc-user user --rc-pass pass
  ```
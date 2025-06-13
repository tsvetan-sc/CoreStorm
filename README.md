# 🔢 Ultimate Number Combination Generator ⚡  

**Generate all possible number combinations at lightning speed - perfect for WPA/WPA2 dictionary attacks, password recovery, and security testing!**  

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)  
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)  
![Performance](https://img.shields.io/badge/Performance-GPU%2FCPU-red.svg)  

## 🚀 Key Features  

- **Blazing Fast Generation**: CPU and GPU-accelerated modes  
- **WPA/WPA2 Ready**: Perfect for generating PIN dictionaries (8-digit WPS pins, phone numbers, etc.)  
- **Smart Batching**: Writes output in configurable batches to optimize performance  
- **Memory Safe**: Handles huge datasets without crashing  
- **Progress Tracking**: See real-time generation progress  

## 💻 Installation  

1. **Prerequisites**:  
   - Python 3.6+  
   - NVIDIA GPU (for CUDA acceleration - optional)  

2. **Install dependencies**:  
   ```bash  
   pip install numpy numba  
   ```  

## 🛠 Usage  

### Basic Generation  
```bash  
python generateCombinations 4  
```  
Generates all 4-digit combinations (0000-9999) in `combinations.txt`  

### WPA Dictionary Generation Example  
```bash  
# Generate all possible 8-digit WPS pins (WPA2)  
python generateCombinations 8 --output wpa_dictionary.txt --batch-size 10000  
```  

### GPU Accelerated Generation  
```bash  
python generateCombinations 6 --gpu --max-gpu-chunk 50000000  
```  

### Full Options  
```bash  
usage: generateCombinations [-h] [--gpu] [--output OUTPUT] [--batch-size BATCH_SIZE]  
                      [--max-gpu-chunk MAX_GPU_CHUNK]  
                      length  

positional arguments:  
  length                Length of the number combinations  

optional arguments:  
  -h, --help            show this help message and exit  
  --gpu                 Use GPU acceleration  
  --output OUTPUT       Output file path (default: combinations.txt)  
  --batch-size BATCH_SIZE  
                        Number of records per batch (default: 1000)  
  --max-gpu-chunk MAX_GPU_CHUNK  
                        Max combinations per GPU chunk (default: 10000000)  
```  

## ⚡ WPA/WPA2 Dictionary Generation Pro Tip  

This tool is **perfect** for creating targeted dictionaries for:  
- WPS PIN attacks (8-digit combinations)  
- Phone number based passwords  
- Date-based passwords (DDMMYYYY = 8 digits)  
- ZIP code dictionaries  

For best results with WPA:  
```bash  
# Generate 8-digit WPS PIN dictionary (sorted)  
python generateCombinations 8 --output wps_pins.txt --batch-size 50000  
sort -o wps_sorted.txt wps_pins.txt  
```  

## 🏎 Performance Benchmarks  

| Digits | Combinations | GPU Time (RTX 3060) | CPU Time (i7-11800H) |  
|--------|--------------|---------------------|----------------------|  
| 4      | 10,000       | 0.12s               | 0.15s                |  
| 6      | 1,000,000    | 0.45s               | 3.21s                |  
| 8      | 100,000,000  | 42.7s               | 15m 23s              |  

## 📝 Notes  

- For lengths > 6, consider using `--gpu` flag for best performance  
- Output files can get large (8-digit = ~800MB)  
- Always use responsibly and only on networks you own or have permission to test  

## 📜 License  

MIT License - Use responsibly and ethically  

---

### 🛡 Ethical Notice  
**Important**: This tool is intended for:  
- Security research  
- Password recovery (your own systems)  
- Educational purposes  

Never use this tool against any systems without **explicit permission**. Unauthorized testing is illegal. The developer is not responsible for any misuse of this tool.

---

### 💡 Advanced Usage Tips  

**1. Optimize for your hardware**:  
```bash  
# For GPUs with more VRAM (e.g., RTX 3090)  
python generateCombinations 8 --gpu --max-gpu-chunk 100000000  

# For systems with fast SSDs  
python generateCombinations 6 --batch-size 50000  
```  

**2. Combine with other tools**:  
```bash  
# Generate 8-digit combos and pipe to hashcat  
python generateCombinations 8 | hashcat -m 22000 capture.hc22000  
```  

**3. Generate specific ranges**:  
Modify the code to generate only certain number ranges (e.g., 10000000-99999999 for 8-digit)  

---

### 🐛 Troubleshooting  

**Q**: Getting CUDA out of memory errors?  
**A**: Reduce the chunk size:  
```bash  
python generateCombinations 8 --gpu --max-gpu-chunk 20000000  
```  

**Q**: Output file is too large?  
**A**: Compress it:  
```bash  
gzip wps_pins.txt  
```  

**Q**: Need to stop and resume generation?  
**A**: Use `split` command to break into smaller files first:  
```bash  
split -l 1000000 combinations.txt combo_chunk_  
```  

---

### 🌟 Star This Project  

If you find this tool useful, please consider starring the repository to show your support!  
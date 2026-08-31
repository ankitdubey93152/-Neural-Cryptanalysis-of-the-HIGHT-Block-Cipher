# Neural Cryptanalysis of the HIGHT Block Cipher

This project reproduces and evaluates deep learning-based cryptanalysis on the full 32-round HIGHT block cipher (ISO/IEC 18033-3). Adapted from Aron Gohr's groundbreaking CRYPTO 2019 study on Speck32/64 (*"Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning"*), this repository systematically tests whether deep neural network architectures can function as effective differential neural distinguishers against HIGHT. HIGHT is a 64-bit block cipher with a 128-bit master key built on an ARX (Addition, Rotation, XOR) Feistel-like structure.

The study evaluates three distinct neural network paradigms—1D Convolutional Neural Networks (1D CNN), 2D Convolutional Neural Networks (2D CNN), and Multi-Layer Perceptrons (MLP)—across three experimental configurations:
- **Experiment A**: Tests single-pair differential distinguisher capability under a fixed plaintext difference $\Delta = \text{0x0000000000000001}$ (least-significant bit flip) vs random differences under fresh 128-bit keys.
- **Experiment B**: Multi-pair distinguisher bundling four ciphertext pairs under a shared key per sample.
- **Experiment C**: Cipher-identification distinguisher discriminating between ciphertexts produced by HIGHT vs the reference cipher SM4 (GB/T 32907-2016) given the identical fixed plaintext difference $\Delta$.

### Block-Size Reconciliation (Experiment C)
HIGHT uses a native 64-bit (8-byte) block size, whereas SM4 uses a native 128-bit (16-byte) block size. For Experiment C, we adopt the recommended approach:
- Plaintexts (8 bytes) are zero-padded to 16 bytes when input to SM4.
- SM4 ciphertexts (16 bytes) are truncated to their first 8 bytes (64 bits).
This ensures both cipher classes provide an identical 64-bit ciphertext pair representation (128 bits total per sample) without introducing artificial zero-padding artifacts into the neural network input vectors.

### Expected Outcome
In alignment with theoretical cryptanalytic expectations for full-round ciphers, models operating on the full 32 rounds of HIGHT exhibit accuracy around 50–52% (equivalent to random guessing). This demonstrates the robust pseudorandomness and differential immunity of full-round HIGHT against standard deep learning distinguishers.

---

### Execution Commands

```bash
pip install -r requirements.txt
python -m pytest tests/ -v

python run_all.py --samples 2000
```

> **Important**: Ensure you are in the repository root directory before running commands:
> ```powershell
> cd -Neural-Cryptanalysis-of-the-HIGHT-Block-Cipher
> ```

| Task | Command |
| :--- | :--- |
| **Run Everything (Experiments A, B, C + Report)** | `python run_all.py --samples 2000` |
| **Run Experiment A (Single Pair)** | `python experiments/run_experiment_a.py --samples 2000` |
| **Run Experiment B (4 Bundled Pairs)** | `python experiments/run_experiment_b.py --samples 2000 --pairs 4` |
| **Run Experiment C (HIGHT vs SM4)** | `python experiments/run_experiment_c.py --samples 2000` |
| **Generate / View Report** | `python generate_report.py` |
| **Run Unit Tests** | `python -m pytest tests/ -v` |

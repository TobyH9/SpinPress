# SpinPress
An attempt at achieving data compression using inspiration from the Ising Model 

## 📌 Overview
This compression scheme introduces a novel approach to data representation by leveraging the Ising model, a mathematical model from statistical mechanics, to encode and reconstruct 2D binary data arrays.

# Low-Temperature Limit Assumption for Compression

In statistical mechanics, systems at equilibrium occupy microstates with probability defined by the **Boltzmann distribution**.

For the purposes of this compression scheme, we will assume that all physical systems that are defined are operating at low enough temperature such that **only the lowest energy microstate has significant probability associated with it**.

This means that the answer to the following question:

> **"What microstate will any physical system in the compression scheme be in at equilibrium?"**

is:

> **"The microstate with the lowest associated energy."**

---

# The 2D Ising Model

We define a **2D lattice** of size $\( N \times N \)$, which has spins $\( s_{ij} \)$, where each spin can take on a value:

$$\[
s_{ij} \in \{-1, +1\}
\]$$

Each spin interacts with its **nearest neighbors** (up, down, left, right), and the energy of a given configuration of spins is defined by the **Ising Hamiltonian**:

$$\[
E = -J \sum_{\langle i,j \rangle} s_{ij} \cdot s_{i'j'}
\]$$

Where:

- The sum is taken over all **nearest neighbor pairs** $\( \langle i,j \rangle \)$
- $\( J \)$ is the interaction constant (assumed $\( J > 0 \)$ for ferromagnetic coupling)
- $\( s_{ij}, s_{i'j'} \in \{-1, +1\} \)$

---

# Microstate Realisation

In our compression scheme, we define the energy of a configuration using the above Hamiltonian and assume the system will settle into the configuration that **minimizes this energy**.

Therefore, for any defined physical system, the microstate it occupies at equilibrium corresponds to the **lowest-energy spin configuration** of the Ising lattice.

---

## ⚙️ How It Works

## Compression Algorithm (Encoding)

**Input:**  
A 2D binary array of size $\( N \times N \)$, where each element is a bit (0 or 1) representing meaningful data.

### Steps:

1. **Map to Ising Spins:**  
   Convert the binary array to Ising spins:
   - Bit `1` $\( \rightarrow \)$ Spin $\( +1 \)$
   - Bit `0` $\( \rightarrow \)$ Spin $\( -1 \)$

2. **Define Target Ground State:**  
   The resulting spin matrix is considered the **desired ground state** of a 2D Ising lattice.

3. **Select Minimal Seed Set:**  
   Identify a minimal subset of spin values and their positions such that:
   - If only these spins are known and all others are random,
   - Then, under energy minimization, the system will evolve into the **original spin matrix**.

4. **Store the Seed Set:**  
   Save:
   - A list of coordinates $\( (i, j) \)$
   - Corresponding spin values $\( s_{ij} \in \{-1, +1\} \)$

**Output:**  
A compressed representation consisting of a small list of $\((i, j, s_{ij})\)$ entries.

---

## Decompression Algorithm (Decoding)

**Input:**  
- The list of seed coordinates and spin values  
- The lattice size $\( N \times N \)$ 
- Knowledge of the energy minimization rules (i.e., Ising Hamiltonian)

### Steps:

1. **Initialize Lattice:**
   - Create an $\( N \times N \)$ spin lattice
   - Set the spins at the given coordinates to their seed values
   - Initialize all other spins randomly (e.g., sampled from $\( \{-1, +1\} \))$

2. **Relax to Equilibrium:**
   - Apply an energy minimization algorithm to evolve the lattice
   - Under the assumption of low temperature, the system will settle into the unique **ground state**

3. **Map Spins Back to Bits:**
   - Convert each spin back to a bit:
     - Spin $\( +1 \rightarrow 1 \)$
     - Spin $\( -1 \rightarrow 0 \)$

**Output:**  
The reconstructed binary array — the original uncompressed data.

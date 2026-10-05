# Translation–rotation projector in `hess2cbovibspec`

Review notes on `_trans_rot_projector` in
[`src/CBOPTvibSpec/utils/hess2cbovibspec.py`](../src/CBOPTvibSpec/utils/hess2cbovibspec.py).
All numerical examples are for formaldehyde, computed from
[`examples/LinRamanSpec/model_data/raman_formaldehyde.hess`](../examples/LinRamanSpec/model_data/raman_formaldehyde.hess),
unless stated otherwise.

Contents

1. [Purpose](#1-purpose)
2. [The transformation in four steps](#2-the-transformation-in-four-steps)
3. [References](#3-references)
4. [Translation and rotation matrices](#4-translation-and-rotation-matrices)
5. [The stacked matrix B](#5-the-stacked-matrix-b)
6. [Orthonormalisation by SVD](#6-orthonormalisation-by-svd)
7. [Structure of the projector P](#7-structure-of-the-projector-p)
8. [Linear molecules](#8-linear-molecules)
9. [Example of the matrix V](#9-example-of-the-matrix-v)
10. [Interpretation of the columns of U](#10-interpretation-of-the-columns-of-u)
11. [How the projector enforces the Eckart conditions](#11-how-the-projector-enforces-the-eckart-conditions)
12. [Summary](#12-summary)

---

## 1. Purpose

`_trans_rot_projector(masses, coords)` builds the matrix $P$ that removes overall
translation and rotation of the molecule from any vector in mass-weighted
Cartesian coordinates. Applied as $P\,H_\mathrm{mw}\,P$, it leaves a Hessian whose
translational and rotational eigenvalues are exactly zero and whose remaining
eigenvectors are pure vibrations.

A free molecule has $3N$ Cartesian degrees of freedom, of which six (five for a
linear molecule) are rigid motions with no restoring force. An exact Hessian has
zero eigenvalues for them. A numerical Hessian does not, because of
integration-grid and convergence noise:

| Formaldehyde | Rigid-motion "frequencies" | Vibrational frequencies vs ORCA |
|---|---|---|
| Raw mass-weighted Hessian | −52 to +109 cm⁻¹ | off by about 0.002 cm⁻¹ |
| Projected Hessian | zero (below 1e-4 cm⁻¹) | within 0.0002 cm⁻¹ |

## 2. The transformation in four steps

**Notation.** Atoms are labelled $a = 1, \dots, N$, with mass $m_a$ (amu).

| Symbol | Meaning | In the code |
|---|---|---|
| $\mathbf R_a$ | equilibrium position of atom $a$ in the frame of the `.hess` file (bohr) | `coords[a]` |
| $\mathbf r_a$ | the same position relative to the centre of mass (step 1) | `com_coords[a]` |
| $\Delta\mathbf x_a$ | Cartesian displacement of atom $a$ from equilibrium: the displaced atom sits at $\mathbf R_a + \Delta\mathbf x_a$ | — |
| $\mathbf q_a$ | mass-weighted displacement, $\mathbf q_a = \sqrt{m_a}\,\Delta\mathbf x_a$ | — |

$\mathbf R_a$ and $\mathbf r_a$ are fixed reference positions; they describe the
geometry at which the Hessian was computed and do not change during a vibration.
$\Delta\mathbf x_a$ and $\mathbf q_a$ are the variables: the Hessian contains
second derivatives of the energy with respect to the $\Delta\mathbf x_a$.
$\Delta\mathbf x_a$ is the same whether measured from $\mathbf R_a$ or from
$\mathbf r_a$, because the two differ by a constant shift.

Stacking the atoms gives $3N$-vectors, ordered (atom 1: $x, y, z$; atom 2:
$x, y, z$; …):

$$
\Delta\mathbf x = \begin{pmatrix} \Delta\mathbf x_1\\ \vdots\\ \Delta\mathbf x_N \end{pmatrix},
\qquad
\mathbf q = \begin{pmatrix} \mathbf q_1\\ \vdots\\ \mathbf q_N \end{pmatrix}
          = M^{1/2}\,\Delta\mathbf x,
\qquad
M = \operatorname{diag}(m_1, m_1, m_1, \dots, m_N, m_N, m_N)
$$

In these coordinates the Hessian becomes
$H_\mathrm{mw} = M^{-1/2} H\, M^{-1/2}$, whose eigenvalues are the squared
angular frequencies.

**Step 1: shift to the centre of mass.**

$$
\mathbf r_a = \mathbf R_a - \frac{\sum_b m_b \mathbf R_b}{\sum_b m_b}
$$

Rotations are defined about the centre of mass, which makes them orthogonal to
the translations in mass-weighted coordinates.

**Step 2: write the six rigid motions in mass-weighted coordinates.**

Each rigid motion is one fixed $3N$-vector in the space of $\mathbf q$, not a
general displacement. To keep them apart from $\mathbf q$, they get their own
names: $\mathbf t_k$ for the translation and $\boldsymbol\rho_k$ for the
rotation along Cartesian axis $\mathbf e_k$, $k \in \{x, y, z\}$. Their
atom-$a$ blocks are $\mathbf t_{k,a}$ and $\boldsymbol\rho_{k,a}$. Each is
obtained from a rigid Cartesian displacement $\Delta\mathbf x_a$ by multiplying
by $\sqrt{m_a}$:

| Rigid motion | Cartesian displacement of atom $a$ | Mass-weighted vector, atom-$a$ block |
|---|---|---|
| unit translation along $\mathbf e_k$ | $\Delta\mathbf x_a = \mathbf e_k$ | $\mathbf t_{k,a} = \sqrt{m_a}\,\mathbf e_k$ |
| infinitesimal rotation about $\mathbf e_k$ through the centre of mass, unit angle | $\Delta\mathbf x_a = \mathbf e_k \times \mathbf r_a$ | $\boldsymbol\rho_{k,a} = \sqrt{m_a}\,(\mathbf e_k \times \mathbf r_a)$ |

A general mass-weighted displacement $\mathbf q$ is a rigid motion only if it is
a linear combination of these six vectors.

They are the columns of the translation matrix $T$, the rotation matrix $R$
(section 4), and the stacked $(3N, 6)$ matrix (section 5)

$$
B = [\,\mathbf t_x \;\; \mathbf t_y \;\; \mathbf t_z \;\;
       \boldsymbol\rho_x \;\; \boldsymbol\rho_y \;\; \boldsymbol\rho_z\,],
$$

whose columns span the translation–rotation subspace.

**Step 3: orthonormalise with an SVD.**

$$
B = U\,S\,V^\mathsf T
$$

The columns of $U$ are an orthonormal basis of the same subspace. Singular
vectors with singular value below $10^{-8}\,\sigma_\mathrm{max}$ are dropped;
the number kept is returned as `n_trans_rot` (section 6).

**Step 4: form the complementary projector.**

$$
P = \mathbb 1 - U\,U^\mathsf T
$$

$U U^\mathsf T$ projects onto the rigid motions, so $P$ projects onto the
vibrational subspace of dimension $3N - n_\mathrm{trans\,rot}$.

**Properties of the result**

- $P$ is symmetric and idempotent, $P^2 = P$.
- $\operatorname{tr} P$ equals the number of vibrations: 6 for formaldehyde, 4 for
  a linear triatomic (both checked in `tests/test_hess2cbovibspec.py`).
- $P\mathbf v = 0$ for any translation or rotation, $P\mathbf v = \mathbf v$ for
  any pure vibration.

**Use in the code**

`hess2cbovibspec` diagonalises $P\,H_\mathrm{mw}\,P$, discards the
`n_trans_rot` zero eigenvalues and converts the rest to frequencies.

## 3. References

The implementation was written from memory of the standard construction, not
from a specific source, and validated numerically against ORCA. The references
below are cited from memory and **have not been checked**; verify the details
before citing them.

- C. Eckart, Phys. Rev. **47**, 552 (1935): the conditions separating rotation
  and translation from vibration.
- E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations* (McGraw-Hill,
  1955): textbook treatment of those conditions and of mass-weighted normal
  coordinates.
- W. H. Miller, N. C. Handy, J. E. Adams, J. Chem. Phys. **72**, 99 (1980): the
  projected force-constant matrix $(1-P)K(1-P)$ in mass-weighted Cartesians,
  which is the form used here.
- J. W. Ochterski, "Vibrational Analysis in Gaussian" (Gaussian Inc. white
  paper, 1999): the closest practical recipe.

Differences from the Ochterski recipe (neither changes the projected subspace):

| | Ochterski | This code |
|---|---|---|
| Rotation axes | principal axes of the inertia tensor | Cartesian axes, $\mathbf e_k \times \mathbf r_a$ |
| Orthogonalisation | Schmidt, dropping vanishing vectors | SVD with singular-value cutoff |
| Hessian used | transformed to the rigid-motion/internal basis, vibrational sub-block diagonalised | $P H_\mathrm{mw} P$ in mass-weighted Cartesians, zero modes discarded |

The relative cutoff $10^{-8}$ is a choice made for this code, not taken from a
reference.

## 4. Translation and rotation matrices

Both matrices have shape $(3N, 3)$. Rows are indexed by (atom $a$, Cartesian
component $j$), columns by the axis $k$ of the rigid motion.

Notation: $m_a$ atomic mass, $M_\mathrm{tot} = \sum_a m_a$,
$\mathbf r_a = (x_a, y_a, z_a)$ centre-of-mass coordinates, $\mathbf e_k$
Cartesian unit vectors.

### Translation matrix

$$
T_{(a,j),k} = \sqrt{m_a}\;\delta_{jk}
\qquad\Longrightarrow\qquad
T^{(a)} = \sqrt{m_a}
\begin{pmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1 \end{pmatrix}
$$

The first column is
$(\sqrt{m_1}, 0, 0, \sqrt{m_2}, 0, 0, \dots)^\mathsf T$. These are Ochterski's
vectors $D_1, D_2, D_3$.

### Rotation matrix

$$
R_{(a,j),k} = \sqrt{m_a}\,(\mathbf e_k \times \mathbf r_a)_j
            = \sqrt{m_a}\sum_l \varepsilon_{jkl}\, r_{a,l}
\qquad\Longrightarrow\qquad
R^{(a)} = \sqrt{m_a}
\begin{pmatrix} 0 & z_a & -y_a\\ -z_a & 0 & x_a\\ y_a & -x_a & 0 \end{pmatrix}
$$

$T^{(a)}$ and $R^{(a)}$ denote the $3\times 3$ blocks of rows belonging to atom
$a$ (not to be confused with the position $\mathbf R_a$). The columns of
$R^{(a)}$ are $\mathbf e_x \times \mathbf r_a$,
$\mathbf e_y \times \mathbf r_a$, $\mathbf e_z \times \mathbf r_a$.

### Ochterski's form of the rotations

He diagonalises the inertia tensor, $I = X\,I'\,X^\mathsf T$, with the principal
axes $\mathbf u_1, \mathbf u_2, \mathbf u_3$ as columns of $X$, and defines
$\mathbf P_a = X^\mathsf T \mathbf r_a$ (coordinates of atom $a$ in the
principal-axis frame). His rotation vectors are

$$
\begin{aligned}
D_{4,(a,j)} &= \sqrt{m_a}\,\big[(P_y)_a X_{j3} - (P_z)_a X_{j2}\big]\\
D_{5,(a,j)} &= \sqrt{m_a}\,\big[(P_z)_a X_{j1} - (P_x)_a X_{j3}\big]\\
D_{6,(a,j)} &= \sqrt{m_a}\,\big[(P_x)_a X_{j2} - (P_y)_a X_{j1}\big]
\end{aligned}
$$

i.e. $\sqrt{m_a}\,(\mathbf u_k \times \mathbf r_a)$, rotations about the
principal axes. The two forms are related by

$$
R_\mathrm{Ochterski} = R\,X
$$

and since $X$ is orthogonal they span the same subspace.

> Recalled, unverified: the printed white paper shows $/\sqrt{m_a}$ rather than
> $\times\sqrt{m_a}$ in these three formulas. Mass-weighted coordinates require
> the multiplication, which is what the code uses.

### Overlaps

$$
T^\mathsf T T = M_\mathrm{tot}\,\mathbb 1_3,
\qquad
T^\mathsf T R = 0,
\qquad
R^\mathsf T R = I,
\quad
I_{kk'} = \sum_a m_a\big(|\mathbf r_a|^2\delta_{kk'} - r_{a,k}\,r_{a,k'}\big)
$$

- $T^\mathsf T R = 0$ because $\sum_a m_a \mathbf r_a = 0$ at the centre of mass.
- The rotations overlap through the inertia tensor $I$. In the principal-axis
  frame $I$ is diagonal, so Ochterski's six vectors are already mutually
  orthogonal and only need normalising.

**Origin of the diagonal value $M_\mathrm{tot}$ in the translation block.** It is
the squared length of a translation column:

$$
(T^\mathsf T T)_{xx} = \sum_a (\sqrt{m_a})^2 = \sum_a m_a = M_\mathrm{tot},
\qquad
(T^\mathsf T T)_{xy} = 0
$$

The off-diagonal elements vanish because different translation columns have
their non-zero entries in different rows. For formaldehyde,
$M_\mathrm{tot} = 1.008 + 12.011 + 1.008 + 15.999 = 30.026$ amu. A uniform unit
shift of all atoms has squared length $M_\mathrm{tot}$ in mass-weighted
coordinates; the analogue for a unit rotation is the moment of inertia about
that axis.

## 5. The stacked matrix B

$B = [\,T \;\; R\,]$ has shape $(3N, 6)$: three translation columns followed by
three rotation columns, one $3\times 6$ block of rows per atom. This is the
`trans_rot` array just before the SVD. With $s_a = \sqrt{m_a}$:

```
                 T_x    T_y    T_z      R_x        R_y        R_z
              ┌                                                       ┐
   atom 1  x  │  s_1     0      0        0       s_1·z_1   −s_1·y_1   │
           y  │   0     s_1     0    −s_1·z_1       0       s_1·x_1   │
           z  │   0      0     s_1    s_1·y_1   −s_1·x_1       0      │
              │                                                       │
   atom 2  x  │  s_2     0      0        0       s_2·z_2   −s_2·y_2   │
B =        y  │   0     s_2     0    −s_2·z_2       0       s_2·x_2   │
           z  │   0      0     s_2    s_2·y_2   −s_2·x_2       0      │
              │                                                       │
     ⋮        │   ⋮      ⋮      ⋮        ⋮          ⋮          ⋮      │
              │                                                       │
   atom N  x  │  s_N     0      0        0       s_N·z_N   −s_N·y_N   │
           y  │   0     s_N     0    −s_N·z_N       0       s_N·x_N   │
           z  │   0      0     s_N    s_N·y_N   −s_N·x_N       0      │
              └                                                       ┘
```

Compact block form, with $[\mathbf r_a]_\times \mathbf v = \mathbf r_a \times \mathbf v$:

$$
B^{(a)} = \sqrt{m_a}\,\big[\;\mathbb 1_3 \;\big|\; -[\mathbf r_a]_\times\;\big],
\qquad
B = \begin{pmatrix} B^{(1)}\\ B^{(2)}\\ \vdots\\ B^{(N)} \end{pmatrix}
$$

- Columns 1–3: every atom moves by the same unit vector, weighted by
  $\sqrt{m_a}$.
- Columns 4–6: atom $a$ moves by $\mathbf e_k \times \mathbf r_a$, weighted by
  $\sqrt{m_a}$. Each rotation column has a zero in the row of its own axis.

For formaldehyde ($N = 4$), $B$ is $12 \times 6$ with rank 6. For a linear
molecule on the $z$ axis all $x_a, y_a$ vanish, the $R_z$ column is zero and the
rank drops to 5.

## 6. Orthonormalisation by SVD

The six columns of $B$ are neither normalised nor all mutually orthogonal. The
thin SVD (`np.linalg.svd(trans_rot, full_matrices=False)`) replaces them by an
orthonormal set spanning the same subspace:

$$
B = U\,S\,V^\mathsf T
$$

- $U$: $(3N, 6)$, orthonormal columns, $U^\mathsf T U = \mathbb 1_6$
- $S$: diagonal, $\sigma_1 \ge \dots \ge \sigma_6 \ge 0$
- $V$: $(6, 6)$, orthogonal

### What S and V are

The overlap matrix of the rigid motions is block diagonal,

$$
B^\mathsf T B =
\begin{pmatrix} M_\mathrm{tot}\,\mathbb 1_3 & 0\\ 0 & I \end{pmatrix}
= V\,S^2\,V^\mathsf T ,
$$

so the SVD is an eigen-decomposition of this matrix:

- Singular values: $\sqrt{M_\mathrm{tot}}$ three times (translations) and
  $\sqrt{I_1}, \sqrt{I_2}, \sqrt{I_3}$ (rotations).
- $V$: in the rotation block it holds the principal axes (Ochterski's $X$). In
  the translation block the singular values are degenerate, so any orthogonal
  $3\times 3$ mixing is allowed.

### What U is

$$
U = B\,V\,S^{-1}
$$

Each column of $U$ is a linear combination of the original six vectors ($BV$),
divided by its own length ($S^{-1}$): the normalised translations
$T_k/\sqrt{M_\mathrm{tot}}$ and the normalised principal-axis rotations
$(RX)_k/\sqrt{I_k}$. These are Ochterski's six vectors after normalisation.

### Rank truncation

$U = BVS^{-1}$ only holds for $\sigma > 0$. For a zero singular value the
returned column of $U$ is an arbitrary unit vector outside the rigid-motion
subspace; keeping it would project out a real vibration. The code keeps only

$$
\sigma_i > 10^{-8}\,\sigma_\mathrm{max}
$$

| System | Columns kept (`n_trans_rot`) |
|---|---|
| Non-linear molecule | 6 |
| Linear molecule | 5 |
| Single atom | 3 |

The cutoff is relative, so it is independent of the units of mass and length.
The masses enter in amu without conversion: rescaling all masses scales every
column of $B$ by the same factor and leaves the subspace unchanged.

### The projector is unambiguous

The freedom in $V$ (mixing of the degenerate translations, ordering by singular
value) changes individual columns of $U$ but not

$$
U\,U^\mathsf T = B\,(B^\mathsf T B)^{+}\,B^\mathsf T ,
$$

which depends only on the subspace spanned by $B$.

### Why SVD rather than Gram–Schmidt or QR

Gram–Schmidt (Ochterski's choice) and QR also orthonormalise, but need a
separate check for a vanishing vector, and a plain QR on a rank-deficient $B$
returns a spurious sixth column. The SVD gives the orthonormal basis and the
rank test together.

## 7. Structure of the projector P

Two different products of $U$ must be kept apart:

| Product | Shape | Value | Meaning |
|---|---|---|---|
| $U^\mathsf T U$ | $(6, 6)$ | $\mathbb 1_6$ | the six columns are orthonormal |
| $U U^\mathsf T$ | $(3N, 3N)$ | dense, rank 6 | projector onto the rigid motions |

Consequently $P = \mathbb 1 - UU^\mathsf T$ is a **dense** $(3N, 3N)$ matrix in
the Cartesian basis; it has **no** zero $6\times 6$ block. For formaldehyde its
upper-left $6\times 6$ block (H and C coordinates) is

```
 0.888  0.010  0.082 -0.133 -0.019  0.149
 0.010  0.374 -0.091  0.005 -0.307 -0.000
 0.082 -0.091  0.871  0.018 -0.006 -0.268
-0.133  0.005  0.018  0.596  0.009  0.034
-0.019 -0.307 -0.006  0.009  0.266 -0.006
 0.149 -0.000 -0.268  0.034 -0.006  0.308
```

The block structure appears only after a change of basis. With
$W = [\,U \;|\; U_\perp\,]$, where the $3N-6$ columns of $U_\perp$ span the
vibrational subspace,

$$
W^\mathsf T P\,W =
\begin{pmatrix} 0_6 & 0\\ 0 & \mathbb 1_{3N-6} \end{pmatrix}
$$

Checked for formaldehyde: six zeros and six ones on the diagonal, off-diagonal
elements below $10^{-15}$.

- **Ochterski** works in that rotated basis and diagonalises only the
  $(3N-6, 3N-6)$ sub-block of the transformed Hessian.
- **The code** diagonalises the full $P\,H_\mathrm{mw}\,P$ and discards the zero
  eigenvalues.

Both give the same frequencies and vibrational eigenvectors. The code's route
returns the eigenvectors directly in mass-weighted Cartesians, the form needed
for transforming the dipole and polarizability derivatives.

## 8. Linear molecules

For a linear molecule the moment of inertia about the molecular axis vanishes,
so one singular value is zero and the corresponding column of $U$ is deleted.
Two refinements:

- **"Exactly zero" holds analytically, not numerically.** If the molecule lies
  along a Cartesian axis, one rotation column of $B$ is identically zero. In a
  general orientation none of the three rotation columns vanishes; they are
  linearly dependent, and the computed $\sigma_6$ is at machine-precision level
  relative to $\sigma_1$. Hence the test $\sigma > 10^{-8}\sigma_\mathrm{max}$
  rather than $\sigma \ne 0$.
- **The deleted column is always the last one**, because singular values are
  returned in descending order. $UU^\mathsf T$ then has rank 5 and
  $\mathbb 1 - UU^\mathsf T$ projects onto the $3N-5$ vibrations.

Numerical check for CO₂ (O–C–O, bond length 2.2 bohr, masses in amu):

| Orientation | $\sigma_6$ (absolute) | $\sigma_6/\sigma_1$ |
|---|---|---|
| Along the $z$ axis | exactly 0 | 0 |
| 1000 random orientations | e.g. 8e-16 | median 2.6e-17, range 4e-19 to 1.6e-16 |

The other singular values are 12.44 (twice, the two non-zero moments of inertia)
and 6.63 (three times, $\sqrt{M_\mathrm{tot}}$). The noise scales with
$\sigma_1 \sim 10$ in amu/bohr units, so the absolute $\sigma_6$ is typically
$10^{-16}$ to $10^{-15}$. The worst ratio observed is about eight orders of
magnitude below the cutoff.

## 9. Example of the matrix V

$V$ for formaldehyde. Rows: the original six rigid motions. Columns: singular
vectors in order of descending singular value.

```
              σ=7.217  σ=6.763  σ=5.480  σ=5.480  σ=5.480  σ=2.519
       T_x  [  0        0        0       -1        0        0      ]
       T_y  [  0        0       -0.9986   0       -0.0535   0      ]
V  =   T_z  [  0        0        0.0535   0       -0.9986   0      ]
       R_x  [ -0.0457  -0.1078   0        0        0        0.9931 ]
       R_y  [  0.9826  -0.1840   0        0        0        0.0252 ]
       R_z  [  0.1800   0.9770   0        0        0        0.1144 ]
```

- **Block structure:** translations and rotations never mix, reflecting
  $T^\mathsf T R = 0$.
- **Translation columns (3–5):** $\sigma^2 = 30.026 = M_\mathrm{tot}$ in amu.
  They are degenerate, so the returned combination ($-T_x$ and a rotation of
  $T_y, T_z$ by about 3°) is arbitrary and has no physical meaning.
- **Rotation columns (1, 2, 6):** the principal axes of inertia, with
  $\sigma^2$ = 52.08, 45.74 and 6.34 amu·bohr².

Diagonalising the inertia tensor directly gives moments 6.34, 45.74, 52.08 and

```
       [  0.9931  -0.1078   0.0457 ]
X  =   [  0.0252  -0.1840  -0.9826 ]
       [  0.1144   0.9770  -0.1800 ]
```

The rotation block of $V$ contains the same vectors in reverse order and with
one sign flipped. The smallest moment belongs to the axis
$(0.993, 0.025, 0.114)$, the C=O bond direction, about which only the two
hydrogens contribute.

Consistency check: $V S^2 V^\mathsf T$ reproduces $B^\mathsf T B$ to $5\times10^{-14}$:

```
        [ 30.026   0       0       0       0       0     ]
        [  0      30.026   0       0       0       0     ]
BᵀB  =  [  0       0      30.026   0       0       0     ]
        [  0       0       0       6.896  -1.272  -4.526 ]
        [  0       0       0      -1.272  51.835   1.008 ]
        [  0       0       0      -4.526   1.008  45.427 ]
```

The lower-right block is the inertia tensor in the Cartesian frame of the
`.hess` file; it is not diagonal because the molecule is not aligned with the
axes.

## 10. Interpretation of the columns of U

$V$ says how to combine the six columns of $B$; $U$ is the result of that
combination in the $3N$-dimensional space of atomic displacements. The columns
of $U$ are the six rigid-body motions themselves, as unit vectors in
mass-weighted Cartesian coordinates.

Formally they are eigenvectors of the other product, $B B^\mathsf T$
($3N \times 3N$):

$$
B B^\mathsf T \mathbf u_i = \sigma_i^2\,\mathbf u_i,
\qquad
\mathbf u_i = \frac{B\,\mathbf v_i}{\sigma_i}
$$

$BB^\mathsf T$ has the same six non-zero eigenvalues as $B^\mathsf T B$, plus
$3N-6$ zeros belonging to the vibrational subspace.

**Translation columns.** For a unit direction $\mathbf n$:

$$
\mathbf u(a) = \sqrt{\frac{m_a}{M_\mathrm{tot}}}\;\mathbf n
$$

As a Cartesian displacement (divide by $\sqrt{m_a}$) every atom moves by
$\mathbf n/\sqrt{M_\mathrm{tot}}$: a rigid shift.

**Rotation columns.** For principal axis $\mathbf u_k$ with moment $I_k$:

$$
\mathbf u(a) = \sqrt{m_a}\;\frac{\mathbf u_k \times \mathbf r_a}{\sqrt{I_k}}
$$

As a Cartesian displacement each atom moves by
$(\mathbf u_k \times \mathbf r_a)/\sqrt{I_k}$: an infinitesimal rigid rotation.

**What projecting onto them measures.** For a mass-weighted displacement
$\mathbf q$ with $\mathbf q_a = \sqrt{m_a}\,\Delta\mathbf x_a$:

$$
\begin{aligned}
\text{translation:}\quad \mathbf u^\mathsf T \mathbf q
  &= \sqrt{M_\mathrm{tot}}\;\big(\mathbf n\cdot\Delta\mathbf R_\mathrm{com}\big)\\
\text{rotation:}\quad \mathbf u^\mathsf T \mathbf q
  &= \frac{1}{\sqrt{I_k}}\;\mathbf u_k\cdot\sum_a m_a\,(\mathbf r_a\times\Delta\mathbf x_a)
   = \sqrt{I_k}\;\Delta\varphi_k
\end{aligned}
$$

The six numbers $U^\mathsf T\mathbf q$ are the centre-of-mass shift and the net
rotation angle contained in $\mathbf q$, in mass-weighted form.

**Relation to the Hessian.** For an exact Hessian the columns of $U$ are its six
eigenvectors with eigenvalue zero. They follow from masses and geometry alone,
which is what allows them to be removed analytically.

## 11. How the projector enforces the Eckart conditions

The projector subtracts from any displacement exactly the rigid shift and rigid
rotation it contains.

**Step 1: the projected vector has no component along $U$.** With
$\mathbf q' = P\mathbf q = \mathbf q - UU^\mathsf T\mathbf q$ and
$U^\mathsf T U = \mathbb 1_6$:

$$
U^\mathsf T\mathbf q' = U^\mathsf T\mathbf q - (U^\mathsf T U)\,U^\mathsf T\mathbf q = 0
$$

**Step 2: $U^\mathsf T\mathbf q' = 0$ is the Eckart conditions.** Since
$B^\mathsf T = V S\,U^\mathsf T$ with $VS$ invertible (after truncation for
linear molecules), $U^\mathsf T\mathbf q' = 0$ is equivalent to
$T^\mathsf T\mathbf q' = 0$ and $R^\mathsf T\mathbf q' = 0$. With
$\mathbf q'_a = \sqrt{m_a}\,\Delta\mathbf x'_a$:

$$
\begin{aligned}
(T^\mathsf T\mathbf q')_k &= \sum_a \sqrt{m_a}\;q'_{a,k}
   = \Big(\sum_a m_a\,\Delta\mathbf x'_a\Big)_k\\
(R^\mathsf T\mathbf q')_k &= \sum_a \sqrt{m_a}\,(\mathbf e_k\times\mathbf r_a)\cdot\mathbf q'_a
   = \Big(\sum_a m_a\,\mathbf r_a\times\Delta\mathbf x'_a\Big)_k
\end{aligned}
$$

using $(\mathbf e_k\times\mathbf r)\cdot\Delta\mathbf x = \mathbf e_k\cdot(\mathbf r\times\Delta\mathbf x)$.
Setting both to zero:

$$
\sum_a m_a\,\Delta\mathbf x'_a = 0 \quad\text{(no centre-of-mass shift)},
\qquad
\sum_a m_a\,\mathbf r_a\times\Delta\mathbf x'_a = 0 \quad\text{(no net rotation)}
$$

**Step 3: what the projector subtracts.** Because translations and rotations are
orthogonal, the rigid-motion projector splits:

$$
U U^\mathsf T = B\,(B^\mathsf T B)^{-1}B^\mathsf T
             = \frac{T\,T^\mathsf T}{M_\mathrm{tot}} + R\,I^{-1}R^\mathsf T
$$

In Cartesian displacements:

$$
\Delta\mathbf x'_a = \Delta\mathbf x_a - \Delta\mathbf R_\mathrm{com} - \Delta\boldsymbol\varphi\times\mathbf r_a
$$

$$
\Delta\mathbf R_\mathrm{com} = \frac{1}{M_\mathrm{tot}}\sum_b m_b\,\Delta\mathbf x_b,
\qquad
\Delta\boldsymbol\varphi = I^{-1}\sum_b m_b\,(\mathbf r_b\times\Delta\mathbf x_b)
$$

Each atom's displacement has the common shift and its share of the rigid
rotation removed. For a linear molecule $I$ is singular and the truncated SVD
amounts to using its pseudo-inverse: the rotation about the molecular axis is
left out.

**Consequence for the normal modes.** Any eigenvector $\mathbf l$ of
$P H_\mathrm{mw} P$ with non-zero eigenvalue $\lambda$ satisfies

$$
\mathbf l = \frac{1}{\lambda}\,P\,H_\mathrm{mw}\,P\,\mathbf l
\quad\Longrightarrow\quad
P\,\mathbf l = \mathbf l
$$

So every vibrational mode returned by `hess2cbovibspec` satisfies the Eckart
conditions exactly, whatever numerical noise the raw Hessian contains. The same
holds for the matrix $M^{-1/2}L$ used to transform the dipole and polarizability
derivatives.

## 12. Summary

- The SVD of $B$ gives the projector onto the translation–rotation subspace,
  $UU^\mathsf T$; its complement $\mathbb 1 - UU^\mathsf T$ projects onto the
  vibrational subspace.
- $V$ is the eigenvector matrix of $B^\mathsf T B$ (total mass and inertia
  tensor); $U$ contains the normalised rigid-body motions and is the eigenvector
  matrix of $BB^\mathsf T$ for its non-zero eigenvalues.
- Linear molecules are handled by truncating $U$: the last singular value
  vanishes (numerically $\sigma_6/\sigma_1 \sim 10^{-17}$) and its column is
  deleted.
- $P$ is dense in Cartesian coordinates and block diagonal only in the basis
  $[\,U \;|\; U_\perp\,]$.
- Applying $P$ enforces the Eckart conditions, so the vibrational modes and the
  transformed property derivatives are free of translation and rotation.

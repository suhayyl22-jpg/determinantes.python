import numpy as np

print("====================================")
print(" T4: DETERMINANTES 3x3 ")
print("====================================\n")

# =====================================================
# 🔹 MÉTODO 1: GAUSS (PASO A PASO)
# =====================================================
def determinante_gauss(A):
    A = A.astype(float)
    n = A.shape[0]
    det = 1
    
    print("Matriz inicial:")
    print(A, "\n")
    
    for i in range(n):
        print(f"--- Paso {i+1} ---")
        
        # Si pivote es 0, intercambiar filas
        if A[i,i] == 0:
            for j in range(i+1, n):
                if A[j,i] != 0:
                    print(f"Intercambio F{i+1} ↔ F{j+1}")
                    A[[i,j]] = A[[j,i]]
                    det *= -1
                    break
        
        pivote = A[i,i]
        print("Pivote:", pivote)
        det *= pivote
        
        # Hacer ceros debajo
        for j in range(i+1, n):
            factor = A[j,i] / pivote
            print(f"F{j+1} = F{j+1} - ({factor})*F{i+1}")
            A[j] = A[j] - factor*A[i]
        
        print("Matriz actual:")
        print(A, "\n")
    
    print("Producto de pivotes =", det)
    return det


# =====================================================
# 🔹 MÉTODO 2: DIAGONALIZACIÓN
# =====================================================
def determinante_diagonal(A):
    print("Matriz triangular (usando triu):")
    U = np.triu(A)
    print(U)
    
    diag = np.diag(U)
    print("Diagonal:", diag)
    
    det = np.prod(diag)
    print("Producto de la diagonal:", det)
    return det


# =====================================================
# 🔹 MÉTODO 3: COFACTORES (EXPANSIÓN)
# =====================================================
def determinante_cofactores(A):
    print("Usando expansión por cofactores")
    det = np.linalg.det(A)
    print("Resultado:", det)
    return det


# =====================================================
# 🔹 EJERCICIOS (3 MATRICES)
# =====================================================

matrices = [
    np.array([[2, 1, 3],
              [0, 4, 5],
              [1, 2, 1]]),

    np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]]),

    np.array([[3, 0, 2],
              [2, 1, 1],
              [1, 0, 1]])
]

# =====================================================
# 🔹 RESOLUCIÓN DE LOS 3 EJERCICIOS
# =====================================================

for idx, M in enumerate(matrices):
    print("====================================")
    print(f" EJERCICIO {idx+1}")
    print("====================================\n")
    
    print("Matriz:")
    print(M, "\n")
    
    # --------- GAUSS ---------
    print("=== MÉTODO DE GAUSS ===")
    det_gauss = determinante_gauss(M.copy())
    print("Determinante por Gauss:", det_gauss, "\n")
    
    # --------- DIAGONAL ---------
    print("=== MÉTODO DE DIAGONALIZACIÓN ===")
    det_diag = determinante_diagonal(M.copy())
    print("Determinante por diagonalización:", det_diag, "\n")
    
    # --------- COFACTORES ---------
    print("=== MÉTODO DE COFACTORES ===")
    det_cof = determinante_cofactores(M.copy())
    print("Determinante por cofactores:", det_cof, "\n")
    
    # --------- VERIFICACIÓN ---------
    print("=== VERIFICACIÓN FINAL ===")
    print("Los tres métodos deben dar el mismo resultado (puede variar por decimales):")
    print("Gauss:", det_gauss)
    print("Diagonal:", det_diag)
    print("Cofactores:", det_cof)
    print("\n\n")

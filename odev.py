import os
import numpy as np
from numpy import linalg as LA
from scipy import linalg
import math
from numpy import array, identity

def determinant(A):
    return linalg.det(A)

def ata_matrix_trace(A):
    return np.trace(np.dot(A.T, A))

def row_norms(A):
    return np.linalg.norm(A, axis=1)

def column_norms(A):
    return np.linalg.norm(A, axis=0)

def frobenius_norm(A):
    return linalg.norm(A, 'fro')

def check_frobenius_norm(A):
    return frobenius_norm(A) == math.sqrt(ata_matrix_trace(A))

def normalize_frobenius(A):
    return A / frobenius_norm(A)

def eigenvalues(A):
    return linalg.eigvals(A)

def spectral_condition_number(A):
    eigenvalues = linalg.eigvals(A)
    return max(eigenvalues) / min(eigenvalues)

def hadamard_condition_number(A):
    return np.linalg.norm(A, np.inf) * np.linalg.norm(np.linalg.inv(A), np.inf)

def Kramer_inverse(A, b):
    detA = np.linalg.det(A)
    if detA == 0:
        print("Sistem çözümsüz")
        return None
    else:
        n = A.shape[0]
        x = np.zeros(n)
        for i in range(n):
            A_i = A.copy()
            A_i[:, i] = b
            detA_i = np.linalg.det(A_i)
            x[i] = detA_i / detA
        return x

def pivot_inverse(A):
    n = A.shape[0]
    I = np.eye(n)
    A_inv = np.copy(I)
    for i in range(n):
        pivot = np.argmax(np.abs(A[i:,i])) + i
        A[[i, pivot], i:] = A[[pivot, i], i:]
        A_inv[[i, pivot], :] = A_inv[[pivot, i], :]
        for j in range(i+1, n):
            m = A[j][i] / A[i][i]
            A[j] -= m * A[i]
            A_inv[j] -= m * A_inv[i]
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            m = A[j][i] / A[i][i]
            A[j] -= m * A[i]
            A_inv[j] -= m * A_inv[i]
    for i in range(n):
        A_inv[i] /= A[i][i]
    return A_inv

def gauss_matrix_inverse(A):
    n = A.shape[0]
    A_inv = np.eye(n)
    for i in range(n):
        pivot = A[i,i]
        A[i,:] = A[i,:] / pivot
        A_inv[i,:] = A_inv[i,:] / pivot
        for j in range(i+1,n):
            factor = A[j,i]
            A[j,:] = A[j,:] - factor * A[i,:]
            A_inv[j,:] = A_inv[j,:] - factor * A_inv[i,:]
    for i in range(n-1,-1,-1):
        for j in range(i-1,-1,-1):
            factor = A[j,i]
            A[j,:] = A[j,:] - factor * A[i,:]
            A_inv[j,:] = A_inv[j,:] - factor * A_inv[i,:]
    return A_inv

def gauss_x_bilinmeyenler(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    for k in range(n-1):
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - m * A[k, k:]
            b[i] = b[i] - m * b[k]
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def gauss_jordan(A, b):
    n = len(A)
    m = len(A[0])
    
    A = array(A)
    I = identity(n)
    AI = A
    AI = concatenate((AI, I), axis=1)
    
    for i in range(n):
        pivot = AI[i][i]
        for j in range(i+1, n):
            if abs(AI[j][i]) > abs(pivot):
                AI[[i,j]] = AI[[j,i]]
                pivot = AI[i][i]
        
        for j in range(i+1, 2*n):
            AI[i][j] /= pivot
        
        for k in range(n):
            if k != i:
                temp = AI[k][i]
                for j in range(i, 2*n):
                    AI[k][j] -= AI[i][j] * temp
    
    x = AI[:, n:]
    return x

def CholeskyFactorHesapla(A):
    L = LA.cholesky(A)
    return L

def ATA_Ters_ModernGauss(ATA):
    ATA_inv = LA.inv(ATA)
    return ATA_inv

def ATA_Ters_Cholesky(ATA):
    L = LA.cholesky(ATA)
    LT = L.T
    ATA_inv = LA.solve(L, LT)
    return ATA_inv

def print_menu():
    print("####### Aşağıdaki menüden bir matris işlemi seçiniz #######")
    print("\t 1. A Determinant hesapla.")
    print("\t 2. ATA Matris izini hesapla.")
    print("\t 3. A Matris satir normlarini hesapla.")
    print("\t 4. A Matris sütun normlarini hesapla.")
    print("\t 5. A Matris Oklid normlarini hesapla.")
    print("\t 6. A N(A) = (iz(ATA))^1/2 esit mi.")
    print("\t 7. A Matrisi Oklid normuna gore normlastir.")
    print("\t 8. A Matrisin ozdegerlerini hesapla.")
    print("\t 9. A Matrisin Spektral (Todd) sart sayisini hesapla ve kararsizligini yorumla.")
    print("\t 10. A Matrissinin Hadamard sart sayisini hesapla ve kararsizligini yorumla.")
    print("\t 11. Kramer kurali ile A Matrisinin tersini hesapla.")
    print("\t 12. Pivotlama ile matris tersi.")
    print("\t 13. Gauss ile matris tersi.")
    print("\t 14. Gauss algoritmasi ile x bilinmeyenler verktorunu hesapla.")
    print("\t 15. Gauss-Jordan yontemi ile x bilinmeyenler verktorunu hesapla.")
    print("\t 16. Modernlestirilmis Gauss Algoritmasi ile x bilinmeyenler vektorunu hesapla.")
    print("\t 17. CholeskyFactorHesapla yontemi ile x bilinmeyenler vektorunu hesapla.")
    print("\t 18. Modernlestirilmis Gauss Algoritmasi ile ATA matrisinin tersini hesapla.")
    print("\t 19. CholeskyFactorHesapla yontemi ile ATA matrisinin tersini hesapla.")


def menu_selection(matrix, vector):
    while True:
        print_menu()
        choice = int(input("Lütfen yapmak istediğiniz işlemi seçiniz: "))
        if choice == 0:
            exit()
        elif choice == 1:
            print(determinant(matrix))
        elif choice == 2:
            print(ata_matrix_trace(vector))
        elif choice == 3:
            print(row_norms(matrix))
        elif choice == 4:
            print(column_norms(matrix))
        elif choice == 5:
            print(frobenius_norm(matrix))
        elif choice == 6:
            print(check_frobenius_norm(matrix))
        elif choice == 7:
            print(normalize_frobenius(matrix))
        elif choice == 8:
            print(eigenvalues(matrix))
        elif choice == 9:
            print(spectral_condition_number(matrix))
        elif choice == 10:
            print(hadamard_condition_number(matrix))
        elif choice == 11:
            print(Kramer_inverse(matrix))
        elif choice == 12:
            print(pivot_inverse(matrix))
        elif choice == 13:
            print(gauss_matrix_inverse(matrix))
        elif choice == 14:
            print(gauss_x_bilinmeyenler(matrix, vector))
        elif choice == 15:
            print(gauss_jordan(matrix, vector))
        elif choice == 16:
            print(gauss_x_bilinmeyenler(matrix, vector))
        elif choice == 17:
            print(CholeskyFactorHesapla(matrix))
        elif choice == 18:
            print(ATA_Ters_ModernGauss(matrix))
        elif choice == 19:
            print(ATA_Ters_Cholesky(matrix))

def read_file_to_np_array(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [x.strip().split(' ') for x in data]
    data = [[float(val) for val in row] for row in data]
    return np.array(data)

def list_txt_files():
    files = [f for f in os.listdir() if f.endswith('.txt')]
    return files

def file_selection():
    global selected_matrix, selected_vector
    files = list_txt_files()
    while True:
        for i, f in enumerate(files):
            print(f"{i+1}. {f}")
        print("0. Çıkış")
        choice = input("Lütfen bir matris seçiniz: ")
        if choice == "0":
            exit()
        else:
            try:
                choice = int(choice)
                if choice > 0 and choice <= len(files):
                    selected_matrix = files[choice-1]
                    print(f"Seçilen matris: {selected_matrix}")
                    break
                else:
                    print("Lütfen geçerli bir seçim yapın.")
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
    while True:
        for i, f in enumerate(files):
            print(f"{i+1}. {f}")
        print("0. Çıkış")
        choice = input("Lütfen bir vektör seçiniz: ")
        if choice == "0":
            exit()
        else:
            try:
                choice = int(choice)
                if choice > 0 and choice <= len(files):
                    selected_vector = files[choice-1]
                    print(f"Seçilen vektör: {selected_vector}")
                    break
                else:
                    print("Lütfen geçerli bir seçim yapın.")
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")

if __name__ == "__main__":
    file_selection()
    matrix = read_file_to_np_array(selected_matrix)
    vector = read_file_to_np_array(selected_vector)

    menu_selection(matrix, vector)
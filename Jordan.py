import numpy as np

# 定义一个函数来计算特征值和特征向量
def eigenvalues_and_vectors(A):
    return np.linalg.eig(A)

def get_rank(A, lambda_val ,t):
    return np.linalg.matrix_rank(np.linalg.matrix_power(A - lambda_val * np.eye(A.shape[0]), t) )

def get_delta(A, lambda_val ,t):
    return (get_rank(A,lambda_val,t-1)-get_rank(A,lambda_val,t)) - (get_rank(A,lambda_val,t)-get_rank(A,lambda_val,t+1))



# 定义一个函数来计算Jordan标准型
def jordan_normal_form(A):
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eigenvalues_and_vectors(A)
    eigenvalues = eigenvalues.real  # 只考虑实数特征值

    # 初始化Jordan块的信息
    # jordan_blocks = {}

    # 对于每个特征值，计算Jordan块
    unique_eigens = []
    # print(eigenvalues)
    for eigen in eigenvalues:
        if len(unique_eigens) == 0:
            unique_eigens.append(eigen)
        else:
            least_diff = 1
            for unique_eigen in unique_eigens:
                if abs(eigen - unique_eigen) < least_diff:
                    least_diff = abs(eigen - unique_eigen)
            if least_diff > 1e-6:
                unique_eigens.append(eigen)
    
    # eigenvalues = np.unique(eigenvalues)
    # print(unique_eigens)
    for lambda_val in unique_eigens:
        m_i = np.sum(eigenvalues == lambda_val) # 代数重数
        rank = [0]
        # r[i]是r_t,也就是rank(A-λI)^t。
        k = None
        # delta = []
        # delta[0]是delta_1
        # k_max = A.shape[0]+2
        # for i in range(k_max):
        #     r.append(np.linalg.matrix_rank(np.linalg.matrix_power(A - lambda_val * np.eye(A.shape[0]), i) ))
        total_n = 0 # 当前各阶jordan子块的阶数之和，其等于阶数×该阶子块个数
        delta_s = [0] # delta[i]是i阶jordan子块的个数
        print(delta_s)
        for i in range(1,m_i+1):
            # print(get_delta(A, lambda_val ,i))
            delta = get_delta(A, lambda_val ,i)
            # print(delta)
            delta_s.append(delta)
            total_n = total_n + delta * i
            if total_n == m_i:
                break
        print(delta_s)


        # print(r)
        # for i in range(1,A.shape[0]+1):
        #     if r[i] == r[i+1]:
        #         k = i
        # for t in range(1,k):
        #     delta.append(r[t-1] + r[t+1] - 2 * r[t])
            
            # print(t, delta)
        # print("lambda:", lambda_val, "delta:", delta)
        # print(lambda_val, delta)

        # 计算特征值的代数重数
    # m_i = np.sum(eigenvalues == lambda_val)

    # 计算Jordan子块的最大阶数
    # k_i = 0
    # while np.linalg.matrix_rank(np.linalg.matrix_power(A - lambda_val * np.eye(A.shape[0]), k_i + 1)  ) == np.linalg.matrix_rank(np.linalg.matrix_power(A - lambda_val * np.eye(A.shape[0]), k_i)):
    #  k_i += 1




    # return m_i, np.unique(eigenvalues)


    # 计算每个特征值的Jordan子块的个数
    # powers = np.linalg.matrix_power(A - lambda_val * np.eye(A.shape[0]), np.arange(1, k_i + 1))
    # r_i = np.array([np.linalg.matrix_rank(powers[i]) for i in range(k_i + 1)])
    # delta_t_i = np.zeros(k_i)
    # for t in range(1, k_i):
    #     delta_t_i[t] = r_i[t - 1] + r_i[t + 1] - 2 * r_i[t]

    # 记录Jordan块的信息
    # jordan_blocks[lambda_val] = {'t': np.arange(1, k_i + 1), 'delta': delta_t_i}

    # 构建Jordan标准型
    # jordan_matrix = np.zeros((A.shape[0], A.shape[0]))
    # for lambda_val, info in jordan_blocks.items():
    #     for t, delta in zip(info['t'], info['delta']):
    #         jordan_block = np.eye(t) * lambda_val
    #         jordan_block[1:, 0:t-1] = 1
    #         jordan_matrix[np.ix_(range(t*m_i), range(t*m_i, (t+1)*m_i))] = jordan_block

    # return jordan_matrix

# 示例矩阵
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
        
# A = np.array([
#     [3, 6, -15],
#     [1, 2, -5],
#     [1, 2, -5],
# ])



# 计算Jordan标准型
jordan_matrix = jordan_normal_form(A)
# print(jordan_matrix)
# print("Jordan Normal Form:")
# print(jordan_matrix)

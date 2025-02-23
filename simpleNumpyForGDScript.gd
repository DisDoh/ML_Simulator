extends Node

var random: RandomNumberGenerator = RandomNumberGenerator.new()

func zeros(matLengthX, matLengthY, matLengthZ=0):
	var matrix = []
	if matLengthZ == 0:
		for x in range(matLengthX):
			matrix.append([])
			for y in range(matLengthY):
				matrix[x].append(0)
	else:
		for x in range(matLengthX):
			matrix.append([])
			for y in range(matLengthY):
				matrix[x].append([])
				for z in range(matLengthZ):
					matrix[x][y].append(0)
	return matrix

func matmul(mat1, mat2):
	if mat1.size() > 0 and mat1[0].size() > 0 and mat1[0][0] is Array:
		if mat1.size() != mat2.size():
			push_error("Erreur: Nombre de batches incompatibles pour la multiplication de matrices 3D")
			return []
		var result = []
		for i in range(mat1.size()):
			result.append(matmul(mat1[i], mat2[i]))
		return result
	else:
		if len(mat1[0]) != len(mat2):
			push_error("Erreur: Dimensions incompatibles pour la multiplication de matrices")
			return []
		var result = zeros(len(mat1), len(mat2[0]))
		for x in range(len(mat1)):
			for y in range(len(mat2[0])):
				for z in range(len(mat2)):
					result[x][y] += mat1[x][z] * mat2[z][y]
		return result

func randn(sizeX, sizeY, sizeZ=0):
	var result = zeros(sizeX, sizeY, sizeZ)
	for x in range(sizeX):
		for y in range(sizeY):
			if sizeZ == 0:
				result[x][y] = random.randfn(0, 1)
			else:
				for z in range(sizeZ):
					result[x][y][z] = random.randfn(0, 1)
	return result

func reshape(mat, newSizeX, newSizeY, newSizeZ=0):
	var elements = []
	if mat.size() == 0:
		push_error("Erreur: matrice vide")
		return []
	if mat[0].size() > 0 and mat[0][0] is Array:
		for x in mat:
			for y in x:
				for z in y:
					elements.append(z)
	else:
		for row in mat:
			elements.append_array(row)

	if newSizeZ == 0:
		if elements.size() != newSizeX * newSizeY:
			push_error("Erreur: Impossible de redimensionner, tailles incompatibles")
			return []
		var newMat = zeros(newSizeX, newSizeY)
		var index = 0
		for x in range(newSizeX):
			for y in range(newSizeY):
				newMat[x][y] = elements[index]
				index += 1
		return newMat
	else:
		if elements.size() != newSizeX * newSizeY * newSizeZ:
			push_error("Erreur: Impossible de redimensionner, tailles incompatibles")
			return []
		var newMat = zeros(newSizeX, newSizeY, newSizeZ)
		var index = 0
		for x in range(newSizeX):
			for y in range(newSizeY):
				for z in range(newSizeZ):
					newMat[x][y][z] = elements[index]
					index += 1
		return newMat

func transpose(mat):
	if mat.size() > 0 and mat[0].size() > 0 and mat[0][0] is Array:
		var transposed = zeros(len(mat[0]), len(mat), len(mat[0][0]))
		for x in range(len(mat)):
			for y in range(len(mat[0])):
				for z in range(len(mat[0][0])):
					transposed[y][x][z] = mat[x][y][z]
		return transposed
	else:
		var transposed = zeros(len(mat[0]), len(mat))
		for x in range(len(mat)):
			for y in range(len(mat[0])):
				transposed[y][x] = mat[x][y]
		return transposed

func identity(size, batch=0):
	if batch == 0:
		var mat = zeros(size, size)
		for i in range(size):
			mat[i][i] = 1
		return mat
	else:
		var mat = zeros(batch, size, size)
		for b in range(batch):
			for i in range(size):
				mat[b][i][i] = 1
		return mat

func print_matrix(mat):
	if mat.size() > 0 and mat[0].size() > 0 and mat[0][0] is Array:
		for i in range(len(mat)):
			print("Slice ", i, ":")
			for row in mat[i]:
				print(row)
	else:
		for row in mat:
			print(row)

func matrices_equal(mat1, mat2, tol=0.0001):
	if mat1.size() != mat2.size():
		return false
	if mat1[0].size() > 0 and mat1[0][0] is Array:
		if len(mat1[0]) != len(mat2[0]) or len(mat1[0][0]) != len(mat2[0][0]):
			return false
		for x in range(len(mat1)):
			for y in range(len(mat1[0])):
				for z in range(len(mat1[0][0])):
					if abs(mat1[x][y][z] - mat2[x][y][z]) > tol:
						return false
		return true
	else:
		if len(mat1[0]) != len(mat2[0]):
			return false
		for x in range(len(mat1)):
			for y in range(len(mat1[0])):
				if abs(mat1[x][y] - mat2[x][y]) > tol:
					return false
		return true

func test_zeros():
	var mat = zeros(2, 3)
	assert(matrices_equal(mat, [[0, 0, 0], [0, 0, 0]]), "Error in zeros() 2D")

func test_zeros_3d():
	var mat = zeros(2, 3, 4)
	assert(mat.size() == 2 and mat[0].size() == 3 and mat[0][0].size() == 4, "Error in zeros() 3D")

func test_matmul():
	var mat1 = [[1, 2], [3, 4]]
	var mat2 = [[2, 0], [1, 2]]
	var expected = [[4, 4], [10, 8]]
	var result = matmul(mat1, mat2)
	assert(matrices_equal(result, expected), "Error in matmul() 2D")

func test_matmul_3d():
	var mat1 = [[[1, 2], [3, 4]], [[2, 0], [1, 2]]]
	var mat2 = [[[2, 0], [1, 2]], [[1, 1], [0, 1]]]
	var expected = [[[4, 4], [10, 8]], [[2, 2], [1, 3]]]
	var result = matmul(mat1, mat2)
	assert(matrices_equal(result, expected), "Error in matmul() 3D")

func test_transpose():
	var mat = [[1, 2, 3], [4, 5, 6]]
	var expected = [[1, 4], [2, 5], [3, 6]]
	var result = transpose(mat)
	assert(matrices_equal(result, expected), "Error in transpose() 2D")

func test_transpose_3d():
	var mat = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
	var expected = [[[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]]]
	var result = transpose(mat)
	assert(matrices_equal(result, expected), "Error in transpose() 3D")

func run_tests():
	print("Running all tests...")
	test_zeros()
	test_zeros_3d()
	test_matmul()
	test_matmul_3d()
	test_transpose()
	test_transpose_3d()
	print("All tests passed! ✅")

func _ready():
	run_tests()

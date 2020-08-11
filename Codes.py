import numpy as np
import re
import math

np.set_printoptions(suppress=True)
# Only for testing
# State_File = './toy_example/State_File'
# Symbol_File = './toy_example/Symbol_File'
# Query_File = './toy_example/Query_File'
#
# State_File = './dev_set/State_File'
# Symbol_File = './dev_set/Symbol_File'
# Query_File = './dev_set/Query_File'

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    # -------------------------------------------------------------------------
    # Step 1: Store the probability for state transition
    # Open the state file
    # Create Matrix for state transition (A)
    with open(State_File, "r") as stateFile:
        # Check the number of states and create a empty matrix
        numOfStates = int(stateFile.readline())
        A = np.zeros(shape=(numOfStates, numOfStates))

        # Store states into an array with its id
        states = []
        for i in range(0, numOfStates):
            states.append(stateFile.readline().rstrip())

        # Store transition times
        for line in stateFile.readlines():
            stateTrans = line.split(" ")
            A[int(stateTrans[0]), int(stateTrans[1])] = int(stateTrans[2])

        # Convert to probabilities with add-1 smoothing
        sumStateMat = A.sum(axis=1)

        # Moidify the probabilities with three rules
        for i in range(numOfStates):
            for j in range(numOfStates):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[j] == "BEGIN":
                    A[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    A[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    A[i, j] = (A[i, j] + 1) / (sumStateMat[i] + numOfStates - 1)

    # -------------------------------------------------------------------------
    # Step 2: Store the probability for symbol emission
    # Open the symbol file
    # Create Matrix for symbol emission
    with open(Symbol_File, "r") as symbolFile:
        # Check the number of symbols and create a empty matrix
        numOfSymbols = int(symbolFile.readline())
        B = np.zeros(shape=(numOfStates, numOfSymbols))

        # Store symbols into an array with its id
        symbols = []
        symbolDic = {}
        for i in range(0, numOfSymbols):
            symbol = symbolFile.readline().rstrip()
            symbols.append(symbol)
            symbolDic[symbol] = i

        # Store emission times
        for line in symbolFile.readlines():
            symbolEmit = line.split(" ")
            B[int(symbolEmit[0]), int(symbolEmit[1])] = int(symbolEmit[2])

        # Convert to probabilities with add-1 smoothing
        sumSymbolMat = B.sum(axis=1)

        for i in range(numOfStates):
            for j in range(numOfSymbols):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[i] == "BEGIN":
                    B[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    B[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    B[i, j] = (B[i, j] + 1) / (sumSymbolMat[i] + numOfSymbols + 1)

    # -------------------------------------------------------------------------
    # Step 3: Parse the query file
    with open(Query_File, "r") as queryFile:
        queryTokens = []
        for line in queryFile:
            line = line.strip()
            # Split the content with the following separators and keep the separators
            content = re.split(r"([ \-,()/& ])", line)

            # Remove empty elements
            while " " in content:
                content.remove(" ")
            while "" in content:
                content.remove("")

            # Convert each of the symbol into symbol ID
            for i in range(len(content)):
                if content[i] in symbols:
                    content[i] = symbolDic[content[i]]
                else:
                    content[i] = "UNK"
            queryTokens.append(content)

    # -------------------------------------------------------------------------
    # Step 4: Implement Viterbi Algorithm
    # Input parameters:
    # 1. Transition Probability:    A
    # 2. Emission Probability:      B
    # 3. Observed symbol sequence:  queryTokens

    finalOut = []
    for query in queryTokens:
        queryLen = len(query)

        # Two matrices to store probabilities and corresponding index
        result = np.zeros(shape=(numOfStates - 2, queryLen + 1))
        index = np.zeros(shape=(numOfStates - 2, queryLen - 1))

        # Calculate the starting probabilities for the first state and observation
        for i in range(numOfStates - 2):
            if query[0] == "UNK":
                result[i][0] = A[-2][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1)
            else:
                result[i][0] = A[-2][i] * B[i][query[0]]

        # Calculate the accumulative probabilities
        # Find the maximum probabilities and record the corresponding index
        for j in range(1, queryLen):
            for i in range(numOfStates - 2):
                opt = []
                for k in range(numOfStates - 2):
                    if query[j] == "UNK":
                        opt.append(result[k][j - 1] * A[k][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1))
                    else:
                        opt.append(result[k][j - 1] * A[k][i] * B[i][query[j]])
                result[i][j] = max(opt)
                index[i][j - 1] = opt.index(max(opt))

        # Calculate the ending probabilities for each state
        for i in range(numOfStates - 2):
            result[i][-1] = result[i][-2] * A[i][-1]

        # Trace back to find the state sequence for the maximum probability
        lastCol = result[:, -1]
        maximumProb = np.amax(lastCol)
        maximumIndex = np.where(lastCol == maximumProb)
        prevState = maximumIndex[0][0]

        # Append the END state first
        output = []
        output.append(numOfStates - 1)
        output.append(prevState)

        for j in range(queryLen - 2, -1, -1):
            prevState = int(index[int(prevState)][j])
            output.append(prevState)

        # Append the BEGIN state, reverse the list and append the log probability
        output.append(numOfStates - 2)
        output = output[::-1]
        output.append(math.log(maximumProb))
        finalOut.append(output)

    return finalOut

# with open("Q1_Output.txt", "w") as f:
#     for item in viterbi_algorithm(State_File, Symbol_File, Query_File):
#         f.write("%s\n" % item)

#print(viterbi_algorithm(State_File, Symbol_File, Query_File))

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):  # do not change the heading of the function
    # -------------------------------------------------------------------------
    # Step 1: Store the probability for state transition
    # Open the state file
    # Create Matrix for state transition (A)
    with open(State_File, "r") as stateFile:
        # Check the number of states and create a empty matrix
        numOfStates = int(stateFile.readline())
        A = np.zeros(shape=(numOfStates, numOfStates))

        # Store states into an array with its id
        states = []
        for i in range(0, numOfStates):
            states.append(stateFile.readline().rstrip())

        # Store transition times
        for line in stateFile.readlines():
            stateTrans = line.split(" ")
            A[int(stateTrans[0]), int(stateTrans[1])] = int(stateTrans[2])

        # Convert to probabilities with add-1 smoothing
        sumStateMat = A.sum(axis=1)

        # Moidify the probabilities with three rules
        for i in range(numOfStates):
            for j in range(numOfStates):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[j] == "BEGIN":
                    A[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    A[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    A[i, j] = (A[i, j] + 1) / (sumStateMat[i] + numOfStates - 1)

    # -------------------------------------------------------------------------
    # Step 2: Store the probability for symbol emission
    # Open the symbol file
    # Create Matrix for symbol emission
    with open(Symbol_File, "r") as symbolFile:
        # Check the number of symbols and create a empty matrix
        numOfSymbols = int(symbolFile.readline())
        B = np.zeros(shape=(numOfStates, numOfSymbols))

        # Store symbols into an array with its id
        symbols = []
        symbolDic = {}
        for i in range(0, numOfSymbols):
            symbol = symbolFile.readline().rstrip()
            symbols.append(symbol)
            symbolDic[symbol] = i

        # Store emission times
        for line in symbolFile.readlines():
            symbolEmit = line.split(" ")
            B[int(symbolEmit[0]), int(symbolEmit[1])] = int(symbolEmit[2])

        # Convert to probabilities with add-1 smoothing
        sumSymbolMat = B.sum(axis=1)

        for i in range(numOfStates):
            for j in range(numOfSymbols):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[i] == "BEGIN":
                    B[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    B[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    B[i, j] = (B[i, j] + 1) / (sumSymbolMat[i] + numOfSymbols + 1)

    # -------------------------------------------------------------------------
    # Step 3: Parse the query file
    with open(Query_File, "r") as queryFile:
        queryTokens = []
        for line in queryFile:
            line = line.strip()
            # Split the content with the following separators and keep the separators
            content = re.split(r"([ \-,()/& ])", line)

            # Remove empty elements
            while " " in content:
                content.remove(" ")
            while "" in content:
                content.remove("")

            # Convert each of the symbol into symbol ID
            for i in range(len(content)):
                if content[i] in symbols:
                    content[i] = symbolDic[content[i]]
                else:
                    content[i] = "UNK"
            queryTokens.append(content)

    # -------------------------------------------------------------------------
    # Step 4: Implement top-k Viterbi Algorithm
    # Input parameters:
    # 1. Transition Probability:    A
    # 2. Emission Probability:      B
    # 3. Observed symbol sequence:  queryTokens

    # print(f'Transition Matrix:\n{A}\n')
    # print(f'Emission Matrix:\n{B}\n')
    # print(f'Query Tokens:\n{queryTokens}\n')

    def changePos(list1, list2):
        for i in range(len(list1) - 2, -1, -1):
            if list1[i] < list2[i]:
                return False
        return True


    finalOut = []
    for query in queryTokens:
        queryLen = len(query)

        # Two matrices to store probabilities and corresponding index
        result = np.zeros(shape=(numOfStates - 2, queryLen + 1, k))
        index = np.zeros(shape=(numOfStates - 2, queryLen - 1, k))

        # Calculate the starting probabilities for the first state and observation
        for i in range(numOfStates - 2):
            if query[0] == "UNK":
                result[i, 0, 0] = A[-2][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1)
            else:
                result[i, 0, 0] = A[-2][i] * B[i][query[0]]

        # Calculate the accumulative probabilities
        # Find the maximum probabilities and record the corresponding index
        for j in range(1, queryLen):
            for i in range(numOfStates - 2):
                opt = []
                for l in range(numOfStates - 2):
                    for n in range(k):
                        if query[j] == "UNK":
                            opt.append(result[l, j - 1, n] * A[l][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1))
                        else:
                            opt.append(result[l, j - 1, n] * A[l][i] * B[i][query[j]])
                sortedIndex = np.argsort(opt)[::-1]
                topkIndex = sortedIndex[:k]
                for m in range(k):
                    result[i, j, m] = opt[topkIndex[m]]
                    index[i, j - 1, m] = topkIndex[m]

        # Calculate the ending probabilities for each state
        for i in range(numOfStates - 2):
            for m in range(k):
                result[i, -1, m] = result[i, -2, m] * A[i][-1]

        # print(result)
        # print(index)

        # Trace back to find the state sequence for the maximum probability
        lastCol = result[:, -1, :]
        lastCol = lastCol.flatten()
        maximumIndex = np.argsort(-lastCol)

        # If the kth possibility is same as k + 1, we derive the sequence of both
        count = k
        while lastCol[maximumIndex[count - 1]] == lastCol[maximumIndex[count]]:
            count += 1

        maxKIndex = maximumIndex[:count]

        klist = []
        for m in maxKIndex:
            # Previous state = m // k, num means the depth
            prevState = int(m // k)
            num = int(m % k)
            prob = result[prevState, -1, num]

            # Append the END state
            output = []
            output.append(numOfStates - 1)
            output.append(prevState)

            for j in range(queryLen - 2, -1, -1):
                pos = index[prevState, j, num]
                prevState = int(pos // k)
                num = int(pos % k)
                output.append(prevState)

            # Append the BEGIN state, reverse the list and append the log probability
            output.append(numOfStates - 2)
            output = output[::-1]
            output.append(math.log(prob))
            klist.append(output)

        # Checkting tie situation:
        for i in range(k - 1, 1, -1):
            # Check whether two probabilities are equal
            if klist[i][-1] == klist[i - 1][-1]:
                if changePos(klist[i - 1], klist[i]):
                    # Swap two elements
                    tmp = klist[i]
                    klist[i] = klist[i - 1]
                    klist[i - 1] = tmp

        # Get the top-k sequence in case of kth possibility is same as k+1th
        for sequence in klist[:k]:
            finalOut.append(sequence)

    return finalOut

#print(top_k_viterbi(State_File, Symbol_File, Query_File, 3))

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    # -------------------------------------------------------------------------
    # Step 1: Store the probability for state transition
    # Open the state file
    # Create Matrix for state transition (A)
    with open(State_File, "r") as stateFile:
        # Check the number of states and create a empty matrix
        numOfStates = int(stateFile.readline())
        A = np.zeros(shape=(numOfStates, numOfStates))

        # Store states into an array with its id
        states = []
        for i in range(0, numOfStates):
            states.append(stateFile.readline().rstrip())

        # Store transition times
        for line in stateFile.readlines():
            stateTrans = line.split(" ")
            A[int(stateTrans[0]), int(stateTrans[1])] = int(stateTrans[2])

        # Convert to probabilities with add-1 smoothing
        sumStateMat = A.sum(axis=1)

        # Moidify the probabilities with three rules
        for i in range(numOfStates):
            for j in range(numOfStates):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[j] == "BEGIN":
                    A[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    A[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    A[i, j] = (A[i, j] + 1) / (sumStateMat[i] + numOfStates - 1)

    # -------------------------------------------------------------------------
    # Step 2: Store the probability for symbol emission
    # Open the symbol file
    # Create Matrix for symbol emission
    with open(Symbol_File, "r") as symbolFile:
        # Check the number of symbols and create a empty matrix
        numOfSymbols = int(symbolFile.readline())
        B = np.zeros(shape=(numOfStates, numOfSymbols))

        # Store symbols into an array with its id
        symbols = []
        symbolDic = {}
        for i in range(0, numOfSymbols):
            symbol = symbolFile.readline().rstrip()
            symbols.append(symbol)
            symbolDic[symbol] = i

        # Store emission times
        for line in symbolFile.readlines():
            symbolEmit = line.split(" ")
            B[int(symbolEmit[0]), int(symbolEmit[1])] = int(symbolEmit[2])

        # Convert to probabilities with add-1 smoothing
        sumSymbolMat = B.sum(axis=1)

        for i in range(numOfStates):
            for j in range(numOfSymbols):
                # Rule 1: For the BEGIN state, there is no transition to it
                if states[i] == "BEGIN":
                    B[i, j] = 0.0
                # Rule 2: For the END states, there is no transition from it
                elif states[i] == "END":
                    B[i, j] = 0.0
                # Rule 3: Add-1 Smoothing
                else:
                    B[i, j] = (B[i, j] + 1) / (sumSymbolMat[i] + numOfSymbols + 1)

    # -------------------------------------------------------------------------
    # Step 3: Parse the query file
    with open(Query_File, "r") as queryFile:
        queryTokens = []
        for line in queryFile:
            line = line.strip()
            # Split the content with the following separators and keep the separators
            content = re.split(r"([ \-,()/& ])", line)

            # Remove empty elements
            while " " in content:
                content.remove(" ")
            while "" in content:
                content.remove("")

            # Convert each of the symbol into symbol ID
            for i in range(len(content)):
                if content[i] in symbols:
                    content[i] = symbolDic[content[i]]
                else:
                    content[i] = "UNK"
            queryTokens.append(content)

    # -------------------------------------------------------------------------
    # Step 4: Implement Viterbi Algorithm
    # Input parameters:
    # 1. Transition Probability:    A
    # 2. Emission Probability:      B
    # 3. Observed symbol sequence:  queryTokens

    finalOut = []
    for query in queryTokens:
        queryLen = len(query)

        # Two matrices to store probabilities and corresponding index
        result = np.zeros(shape=(numOfStates - 2, queryLen + 1))
        index = np.zeros(shape=(numOfStates - 2, queryLen - 1))

        # Calculate the starting probabilities for the first state and observation
        for i in range(numOfStates - 2):
            if query[0] == "UNK":
                result[i][0] = A[-2][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1)
            else:
                result[i][0] = A[-2][i] * B[i][query[0]]

        # Calculate the accumulative probabilities
        # Find the maximum probabilities and record the corresponding index
        for j in range(1, queryLen):
            for i in range(numOfStates - 2):
                opt = []
                for k in range(numOfStates - 2):
                    if query[j] == "UNK":
                        opt.append(result[k][j - 1] * A[k][i] * 1 / (sumSymbolMat[i] + numOfSymbols + 1))
                    else:
                        opt.append(result[k][j - 1] * A[k][i] * B[i][query[j]])
                result[i][j] = max(opt)
                index[i][j - 1] = opt.index(max(opt))

        # Calculate the ending probabilities for each state
        for i in range(numOfStates - 2):
            result[i][-1] = result[i][-2] * A[i][-1]

        # Trace back to find the state sequence for the maximum probability
        lastCol = result[:, -1]
        maximumProb = np.amax(lastCol)
        maximumIndex = np.where(lastCol == maximumProb)
        prevState = maximumIndex[0][0]

        # Append the END state first
        output = []
        output.append(numOfStates - 1)
        output.append(prevState)

        for j in range(queryLen - 2, -1, -1):
            prevState = int(index[int(prevState)][j])
            output.append(prevState)

        # Append the BEGIN state, reverse the list and append the log probability
        output.append(numOfStates - 2)
        output = output[::-1]
        output.append(math.log(maximumProb))
        finalOut.append(output)

    return finalOut

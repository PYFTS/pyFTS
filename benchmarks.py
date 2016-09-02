# Erro quadrático médio
def rmse(predictions,targets):
    return np.sqrt(np.mean((predictions-targets)**2))

# Erro Percentual médio
def mape(predictions,targets):
    return np.mean(abs(predictions-targets)/predictions)
    
def plotComparedSeries(original,fts,title):
	fig = plt.figure(figsize=[20,6])
	ax = fig.add_subplot(111)
	predicted = [fts.predict(xx) for xx in original]
	error = rmse(original,predicted)
	ax.plot(original,color='b',label="Original")
	ax.plot(predicted,color='r',label="Predicted")
	handles0, labels0 = ax.get_legend_handles_labels()
	ax.legend(handles0,labels0)
	ax.set_title(title)
	ax.set_ylabel('F(T)')
	ax.set_xlabel('T')
	ax.set_xlim([0,len(original)])
	ax.set_ylim([min(original),max(original)])
	
def plotCompared(original,predicted,labels,title):
	fig = plt.figure(figsize=[13,6])
	ax = fig.add_subplot(111)
	ax.plot(original,color='k',label="Original")
	for c in range(0,len(predicted)):
		ax.plot(predicted[c],label=labels[c])
	handles0, labels0 = ax.get_legend_handles_labels()
	ax.legend(handles0,labels0)
	ax.set_title(title)
	ax.set_ylabel('F(T)')
	ax.set_xlabel('T')
	ax.set_xlim([0,len(original)])
	ax.set_ylim([min(original),max(original)])
	
def SelecaoKFold_MenorRMSE(original,parameters,modelo):
	nfolds = 5
	ret = []
	errors = np.array([[0 for k in parameters] for z in np.arange(0,nfolds)])
	predicted_best = []
	print("Série Original")
	fig = plt.figure(figsize=[18,10])
	fig.suptitle("Comparação de modelos ")
	ax0 = fig.add_axes([0, 0.5, 0.65, 0.45]) #left, bottom, width, height
	ax0.set_xlim([0,len(original)])
	ax0.set_ylim([min(original),max(original)])
	ax0.set_title('Série Temporal')
	ax0.set_ylabel('F(T)')
	ax0.set_xlabel('T')
	ax0.plot(original,label="Original")
	min_rmse_fold = 100000.0
	best = None
	fc = 0 #Fold count
	kf = KFold(len(original), n_folds=nfolds)
	for train_ix, test_ix in kf:
		train = original[train_ix]
		test = original[test_ix]
		min_rmse = 100000.0
		best_fold = None
		predicted_best_fold = []
		errors_fold = []
		pc = 0 #Parameter count
		for p in parameters:
			sets = GridPartitionerTrimf(train,p)
			fts = modelo(str(p)+ " particoes")
			fts.learn(train,sets)
			predicted = [fts.predict(xx) for xx in test]
			error = rmse(np.array(predicted),np.array(test))
			errors_fold.append(error)
			print(fc, p, error)
			errors[fc,pc] = error
			if error < min_rmse:
				min_rmse = error
				best_fold = fts
				predicted_best_fold = predicted
			pc = pc + 1
		predicted_best_fold = [best_fold.predict(xx) for xx in original]
		ax0.plot(predicted_best_fold,label=best_fold.name)
		if np.mean(errors_fold) < min_rmse_fold:
			min_rmse_fold = np.mean(errors)
			best = best_fold
			predicted_best = predicted_best_fold 
		fc = fc + 1
	handles0, labels0 = ax0.get_legend_handles_labels()
	ax0.legend(handles0, labels0)
	ax1 = Axes3D(fig, rect=[0.7, 0.5, 0.3, 0.45], elev=30, azim=144)
	#ax1 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
	ax1.set_title('Comparação dos Erros Quadráticos Médios')
	ax1.set_zlabel('RMSE')
	ax1.set_xlabel('K-fold')
	ax1.set_ylabel('Partições')
	X,Y = np.meshgrid(np.arange(0,nfolds),parameters)
	surf = ax1.plot_surface(X, Y, errors.T, rstride=1, cstride=1, antialiased=True)
	ret.append(best)
	ret.append(predicted_best)

    # Modelo diferencial
	print("\nSérie Diferencial")
	errors = np.array([[0 for k in parameters] for z in np.arange(0,nfolds)])
	predictedd_best = []
	ax2 = fig.add_axes([0, 0, 0.65, 0.45]) #left, bottom, width, height
	ax2.set_xlim([0,len(original)])
	ax2.set_ylim([min(original),max(original)])
	ax2.set_title('Série Temporal')
	ax2.set_ylabel('F(T)')
	ax2.set_xlabel('T')
	ax2.plot(original,label="Original")
	min_rmse = 100000.0
	min_rmse_fold = 100000.0
	bestd = None
	fc = 0
	diff = diferencas(original)       
	kf = KFold(len(original), n_folds=nfolds)
	for train_ix, test_ix in kf:
		train = diff[train_ix]
		test = diff[test_ix]
		min_rmse = 100000.0
		best_fold = None
		predicted_best_fold = []
		errors_fold = []
		pc = 0
		for p in parameters:
			sets = GridPartitionerTrimf(train,p)
			fts = modelo(str(p)+ " particoes")
			fts.learn(train,sets)
			predicted = [fts.predictDiff(test,xx) for xx in np.arange(len(test))]
			error = rmse(np.array(predicted),np.array(test))
			print(fc, p,error)
			errors[fc,pc] = error
			errors_fold.append(error)
			if error < min_rmse:
				min_rmse = error
				best_fold = fts
			pc = pc + 1
		predicted_best_fold = [best_fold.predictDiff(original, xx) for xx in np.arange(len(original))]
		ax2.plot(predicted_best_fold,label=best_fold.name)
		if np.mean(errors_fold) < min_rmse_fold:
			min_rmse_fold = np.mean(errors)
			best = best_fold
			predicted_best = predicted_best_fold
		fc = fc + 1
	handles0, labels0 = ax2.get_legend_handles_labels()
	ax2.legend(handles0, labels0)
	ax3 = Axes3D(fig, rect=[0.7, 0, 0.3, 0.45], elev=30, azim=144)
	#ax1 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
	ax3.set_title('Comparação dos Erros Quadráticos Médios')
	ax3.set_zlabel('RMSE')
	ax3.set_xlabel('K-fold')
	ax3.set_ylabel('Partições')
	X,Y = np.meshgrid(np.arange(0,nfolds),parameters)
	surf = ax3.plot_surface(X, Y, errors.T, rstride=1, cstride=1, antialiased=True)
	ret.append(best)
	ret.append(predicted_best)
	return ret
	
def SelecaoSimples_MenorRMSE(original,parameters,modelo):
	ret = []
	errors = []
	predicted_best = []
	print("Série Original")
	fig = plt.figure(figsize=[20,12])
	fig.suptitle("Comparação de modelos ")
	ax0 = fig.add_axes([0, 0.5, 0.65, 0.45]) #left, bottom, width, height
	ax0.set_xlim([0,len(original)])
	ax0.set_ylim([min(original),max(original)])
	ax0.set_title('Série Temporal')
	ax0.set_ylabel('F(T)')
	ax0.set_xlabel('T')
	ax0.plot(original,label="Original")
	min_rmse = 100000.0
	best = None
	for p in parameters:
		sets = GridPartitionerTrimf(original,p)
		fts = modelo(str(p)+ " particoes")
		fts.learn(original,sets)
		predicted = [fts.predict(xx) for xx in original]
		ax0.plot(predicted,label=fts.name)
		error = rmse(np.array(predicted),np.array(original))
		print(p,error)
		errors.append(error)
		if error < min_rmse:
			min_rmse = error
			best = fts
			predicted_best = predicted
	handles0, labels0 = ax0.get_legend_handles_labels()
	ax0.legend(handles0, labels0)
	ax1 = fig.add_axes([0.7, 0.5, 0.3, 0.45]) #left, bottom, width, height
	ax1.set_title('Comparação dos Erros Quadráticos Médios')
	ax1.set_ylabel('RMSE')
	ax1.set_xlabel('Quantidade de Partições')
	ax1.set_xlim([min(parameters),max(parameters)])
	ax1.plot(parameters,errors)
	ret.append(best)
	ret.append(predicted_best)
    # Modelo diferencial
	print("\nSérie Diferencial")
	errors = []
	predictedd_best = []
	ax2 = fig.add_axes([0, 0, 0.65, 0.45]) #left, bottom, width, height
	ax2.set_xlim([0,len(original)])
	ax2.set_ylim([min(original),max(original)])
	ax2.set_title('Série Temporal')
	ax2.set_ylabel('F(T)')
	ax2.set_xlabel('T')
	ax2.plot(original,label="Original")
	min_rmse = 100000.0
	bestd = None
	for p in parameters:
		sets = GridPartitionerTrimf(diferencas(original),p)
		fts = modelo(str(p)+ " particoes")
		fts.learn(diferencas(original),sets)
		predicted = [fts.predictDiff(original, xx) for xx in range(1,len(original))]
		predicted.insert(0,original[0])
		ax2.plot(predicted,label=fts.name)
		error = rmse(np.array(predicted),np.array(original))
		print(p,error)
		errors.append(error)
		if error < min_rmse:
			min_rmse = error
			bestd = fts
			predictedd_best = predicted
	handles0, labels0 = ax2.get_legend_handles_labels()
	ax2.legend(handles0, labels0)
	ax3 = fig.add_axes([0.7, 0, 0.3, 0.45]) #left, bottom, width, height
	ax3.set_title('Comparação dos Erros Quadráticos Médios')
	ax3.set_ylabel('RMSE')
	ax3.set_xlabel('Quantidade de Partições')
	ax3.set_xlim([min(parameters),max(parameters)])
	ax3.plot(parameters,errors)
	ret.append(bestd)
	ret.append(predictedd_best)
	return ret
	
def compareModelsPlot(original,models_fo,models_ho):
    fig = plt.figure(figsize=[13,6])
    fig.suptitle("Comparação de modelos ")
    ax0 = fig.add_axes([0, 0, 1, 1]) #left, bottom, width, height
    rows = []
    for model in models_fo:
        fts = model["model"]
        ax0.plot(model["predicted"], label=model["name"])
    for model in models_ho:
        fts = model["model"]
        ax0.plot(model["predicted"], label=model["name"])
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0)
    
def compareModelsTable(original,models_fo,models_ho):
    fig = plt.figure(figsize=[12,4])
    fig.suptitle("Comparação de modelos ")
    columns = ['Modelo','Ordem','Partições','RMSE','MAPE (%)']
    rows = []
    for model in models_fo:
        fts = model["model"]
        error_r = rmse(model["predicted"],original)
        error_m = round(mape(model["predicted"],original)*100,2)
        rows.append([model["name"],fts.order,len(fts.sets),error_r,error_m])
    for model in models_ho:
        fts = model["model"]
        error_r = rmse(model["predicted"][fts.order:],original[fts.order:])
        error_m = round(mape(model["predicted"][fts.order:],original[fts.order:])*100,2)
        rows.append([model["name"],fts.order,len(fts.sets),error_r,error_m])
    ax1 = fig.add_axes([0, 0, 1, 1]) #left, bottom, width, height
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.table(cellText=rows,
                      colLabels=columns,
                      cellLoc='center',
                      bbox=[0,0,1,1])
    sup = "\\begin{tabular}{"
    header = ""
    body = ""
    footer = ""

    for c in columns:
        sup = sup + "|c"
        if len(header) > 0:
            header = header + " & "
        header = header + "\\textbf{" + c + "} "
    sup = sup + "|} \\hline\n"
    header = header + "\\\\ \\hline \n"    
    
    for r in rows:
        lin = ""
        for c in r:
            if len(lin) > 0:
                lin = lin + " & "
            lin = lin + str(c)
        
        body = body + lin + "\\\\ \\hline \n" 
        
    return sup + header + body + "\\end{tabular}"

def HOSelecaoSimples_MenorRMSE(original,parameters,orders):
	ret = []
	errors = np.array([[0 for k in range(len(parameters))] for kk in range(len(orders))])
	predicted_best = []
	print("Série Original")
	fig = plt.figure(figsize=[20,12])
	fig.suptitle("Comparação de modelos ")
	ax0 = fig.add_axes([0, 0.5, 0.6, 0.45]) #left, bottom, width, height
	ax0.set_xlim([0,len(original)])
	ax0.set_ylim([min(original),max(original)])
	ax0.set_title('Série Temporal')
	ax0.set_ylabel('F(T)')
	ax0.set_xlabel('T')
	ax0.plot(original,label="Original")
	min_rmse = 100000.0
	best = None
	pc = 0
	for p in parameters:
		oc = 0
		for o in orders:
			sets = GridPartitionerTrimf(original,p)
			fts = HighOrderFTS(o,"k = " + str(p)+ " w = " + str(o))
			fts.learn(original,sets)
			predicted = [fts.predict(original, xx) for xx in range(o,len(original))]
			error = rmse(np.array(predicted),np.array(original[o:]))
			for kk in range(o):
				predicted.insert(0,None)
			ax0.plot(predicted,label=fts.name)
			print(o,p,error)
			errors[oc,pc] = error
			if error < min_rmse:
				min_rmse = error
				best = fts
				predicted_best = predicted
			oc = oc + 1
		pc = pc + 1
		handles0, labels0 = ax0.get_legend_handles_labels()
	ax0.legend(handles0, labels0)
	ax1 = Axes3D(fig, rect=[0.6, 0.5, 0.45, 0.45], elev=30, azim=144)
	#ax1 = fig.add_axes([0.6, 0.5, 0.45, 0.45], projection='3d')
	ax1.set_title('Comparação dos Erros Quadráticos Médios por tamanho da janela')
	ax1.set_ylabel('RMSE')
	ax1.set_xlabel('Quantidade de Partições')
	ax1.set_zlabel('W')
	X,Y = np.meshgrid(parameters,orders)
	surf = ax1.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
	ret.append(best)
	ret.append(predicted_best)

    # Modelo diferencial
	print("\nSérie Diferencial")
	errors = np.array([[0 for k in range(len(parameters))] for kk in range(len(orders))])
	predictedd_best = []
	ax2 = fig.add_axes([0, 0, 0.6, 0.45]) #left, bottom, width, height
	ax2.set_xlim([0,len(original)])
	ax2.set_ylim([min(original),max(original)])
	ax2.set_title('Série Temporal')
	ax2.set_ylabel('F(T)')
	ax2.set_xlabel('T')
	ax2.plot(original,label="Original")
	min_rmse = 100000.0
	bestd = None
	pc = 0
	for p in parameters:
		oc = 0
		for o in orders:
			sets = GridPartitionerTrimf(diferencas(original),p)
			fts = HighOrderFTS(o,"k = " + str(p)+ " w = " + str(o))
			fts.learn(original,sets)
			predicted = [fts.predictDiff(original, xx) for xx in range(o,len(original))]
			error = rmse(np.array(predicted),np.array(original[o:]))
			for kk in range(o):
				predicted.insert(0,None)
			ax2.plot(predicted,label=fts.name)
			print(o,p,error)
			errors[oc,pc] = error
			if error < min_rmse:
				min_rmse = error
				bestd = fts
				predictedd_best = predicted
			oc = oc + 1
		pc = pc + 1
	handles0, labels0 = ax2.get_legend_handles_labels()
	ax2.legend(handles0, labels0)
	ax3 = Axes3D(fig, rect=[0.6, 0.0, 0.45, 0.45], elev=30, azim=144)
	#ax3 = fig.add_axes([0.6, 0.0, 0.45, 0.45], projection='3d')
	ax3.set_title('Comparação dos Erros Quadráticos Médios')
	ax3.set_ylabel('RMSE')
	ax3.set_xlabel('Quantidade de Partições')
	ax3.set_zlabel('W')
	X,Y = np.meshgrid(parameters,orders)
	surf = ax3.plot_surface(X, Y, errors, rstride=1, cstride=1, antialiased=True)
	ret.append(bestd)
	ret.append(predictedd_best)
	return ret

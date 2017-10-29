from pylab import *
svm_acc = [.79,.80,.80,.78,.79,.79,.77,.77,.81,.80]

rnn_acc = [.757,.756,.770,.750,.760,.755,.753,.747,.791,.780]

time = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q", "2017_2q"]

time_num = list(range(len(time)))

plt.xlabel('Time Period')
plt.ylabel('Accuracy')
plt.title('Prediction Accuracy by Time Period')
plt.xticks(time_num, time, rotation=45)
plt.plot(time_num, svm_acc, color="green", linestyle="--", linewidth=3, label="SVM")
plt.plot(time_num, rnn_acc, color="red", linestyle="-", linewidth=3, label="RNN")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


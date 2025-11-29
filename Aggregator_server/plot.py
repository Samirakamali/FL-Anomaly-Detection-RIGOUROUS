def perform_testing(server,loss_glob, dataset_train, dataset_test, all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test):
    acc_train, loss_train, _, _ = server.test(dataset_train)
    acc_test, loss_test, precision, f1 = server.test(dataset_test)
    print("Testing accuracy: {:.2f}%".format(acc_test))
    print('Testing precision {:.2f}%'.format(precision))
    print('Testing f1 {:.2f}%'.format(f1))
    all_acc_train.append(acc_train)
    all_acc_test.append(acc_test)
    all_loss_glob.append(loss_glob)
    all_loss_train.append(loss_train)
    all_loss_test.append(loss_test)
    return all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test, acc_test, precision, f1
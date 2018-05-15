import matplotlib.pyplot as plt
import numpy as np
import inversion_compare

import pdb

if __name__ == "__main__":

    top_path = "eurosys-eval/inversion_results/badVictimModel_"
    csv = ".csv"

    width = 0.25
    ind = np.arange(4)

    means_no_dp = np.zeros(4)
    std_no_dp = np.zeros(4)

    means_e1 = np.zeros(4)
    std_e1 = np.zeros(4)

    means_e5 = np.zeros(4)
    std_e5 = np.zeros(4)

    # means_no_dp[0] = inversion_compare.compare(top_path + str(0) +
    #                                             "b_1" + csv)

    # means_e1[0] = inversion_compare.compare(top_path + str(0) +
    #                                             "b_1e_1" + csv)
    

    # means_e5[0] = inversion_compare.compare(top_path + str(0) +
    #                                             "b_5e_1" + csv)
    
    
    # std_no_dp[0] = 0
    # std_e1[0] = 0
    # std_e5[0] = 0

    for by in range(4):

        # NO DP
        data = np.zeros(3)

        for i in range(3):

            data[i] = inversion_compare.compare(top_path + str(by) +
                                                "b_" + str(i + 1) + csv)

        means_no_dp[by] = np.mean(data)
        std_no_dp[by] = np.std(data)

        # 1-DP
        data = np.zeros(3)

        for i in range(3):

            data[i] = inversion_compare.compare(top_path + str(by) +
                                                "b_1e_" + str(i + 1) + csv)

        means_e1[by] = np.mean(data)
        std_e1[by] = np.std(data)

        # 5-DP
        data = np.zeros(3)

        for i in range(3):

            data[i] = inversion_compare.compare(top_path + str(by) +
                                                "b_5e_" + str(i+1) + ".csv")

        means_e5[by] = np.mean(data)
        std_e5[by] = np.std(data)

    fig, ax = plt.subplots()

    rects1 = plt.bar([p for p in ind],
                     means_e1, width,
                     color='red',
                     yerr=std_e1,
                     label=r'$\varepsilon$ = 1')

    rects2 = plt.bar([p + width for p in ind],
                     means_e5, width,
                     color='orange',
                     yerr=std_e5,
                     label=r'$\varepsilon$ = 5')

    rects3 = plt.bar([p + 2 * width for p in ind],
                     means_no_dp, width,
                     color='green',
                     yerr=std_no_dp,
                     label='No Privacy')

    plt.ylabel("Reconstuction Error", fontsize=18)
    plt.xlabel("# of bystanders (k-2)", fontsize=18)

    plt.xticks(ind + width, ('0', '1', '2', '3'))
    plt.yticks()
    plt.legend(loc='best', ncol=3, fontsize=18)

    axes = plt.gca()
    axes.set_ylim([0, 1])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.show()

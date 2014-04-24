from optparse import OptionParser

class argParse:
    def parseArg(self):
        op = OptionParser()
        op.add_option('--report', action='store_true', dest='print_report', help='Print a detailed classification report.')
        op.add_option('--chi2_select', action='store', type='int', dest='select_chi2', help='Select some number of features using a chi-squared test')
        op.add_option('--confusion_matrix', action='store_true', dest='print_cm', help='Print the confusion matrix.')
        op.add_option('--top10', action='store_true', dest='print_top10', help='Print ten most discriminative terms per class for every classifier.')
        op.add_option('--all_categories', action='store_true', dest='all_categories', help='Whether to use all categories or not.')
        op.add_option('--use_hashing', action='store_true', help='Use a hashing vectorizer.')
        op.add_option('--n_features', action='store', type=int, default=2 ** 16, help='n_features when using the hashing vectorizer.')
        op.add_option('--filtered', action='store_true', help='Remove newsgroup information that is easily overfit: headers, signatures, and quoting.')
        
        (opts, args) = op.parse_args()
        
        if len(args) > 0:
            op.error('this script does not take any argument!')
            sys.exit(1)
        
        print __doc__
        op.print_help()
        print 
        
        return opts, args
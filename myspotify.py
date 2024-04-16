import argparse
import sys
import category
import byuser
import bysong

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--recomend',  action = 'store', choices=['category', 'byuser', 'bysong'],
                        help='Choose type of recomendation', type = str, default = None)   
        parser.add_argument('-p', '--parametr',  action = 'store',
                        help='Add parameter for recomendation', type = str, default = None)          
        args = parser.parse_args(sys.argv[1:])
        if args.recomend == 'category':
            category.category(args.parametr)
        elif args.recomend == 'byuser':
            byuser.byuser(args.parametr)
        elif args.recomend == 'bysong':
            bysong.bysong(args.parametr)
    except Exception as e:
        print(e)      

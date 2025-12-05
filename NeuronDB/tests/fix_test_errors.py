#!/usr/bin/env python3
"""
Script to fix error handling in test files.
Replaces silent exception handling with proper error verification.
"""

import re
import os
import sys
from pathlib import Path

def fix_advance_error_test(content, algorithm_name):
    """Fix error test patterns in advance files"""
    
    # Pattern 1: Simple error test with NULL
    pattern1 = r'(DO \$\$\s+BEGIN\s+BEGIN\s+PERFORM neurondb\.train\([\'"]' + re.escape(algorithm_name) + r'[\'"],[\'"]missing_table[\'"],[\'"]features[\'"],[\'"]label[\'"],[\'"]\{\}[\'"]::jsonb\);\s+RAISE EXCEPTION [\'"]FAIL: expected error for missing table[\'"];\s+)EXCEPTION WHEN OTHERS THEN\s+NULL;\s+-- Error handled correctly\s+NULL;\s+END;\s+END\$\$;)'
    
    replacement1 = r'''\1EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for missing table but no error occurred';
	END IF;
END$$;'''
    
    # More general pattern - this is complex, so we'll do manual fixes
    return content

def fix_negative_test(content):
    """Fix negative test files"""
    # Change ON_ERROR_STOP off to on
    content = re.sub(r'\\set ON_ERROR_STOP off', r'\\set ON_ERROR_STOP on', content)
    
    # Fix simple exception patterns
    # Pattern: DO $$ BEGIN ... EXCEPTION WHEN OTHERS THEN END $$;
    pattern = r'(DO \$\$\s+BEGIN\s+PERFORM neurondb\.train\([^)]+\);\s+RAISE EXCEPTION [\'"]Should have failed[\'"];\s+)EXCEPTION WHEN OTHERS THEN\s+END \$\$;)'
    
    replacement = r'''\1EXCEPTION WHEN OTHERS THEN
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error but no error occurred';
	END IF;
END$$;'''
    
    # This is too complex for regex - manual fixes needed
    return content

if __name__ == '__main__':
    print("This script provides helper functions.")
    print("Manual fixes are recommended for accuracy.")
    sys.exit(0)


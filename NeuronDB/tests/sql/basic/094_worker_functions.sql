-- Test background worker management functions
SELECT proname, pronargs FROM pg_proc 
WHERE proname LIKE 'neuran%' 
ORDER BY proname;

-- Test manual worker execution (should not crash)
SELECT neuranq_run_once();
SELECT neuranmon_sample();


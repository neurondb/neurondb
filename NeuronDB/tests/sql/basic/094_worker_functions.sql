-- Test background worker management functions
SELECT proname, pronargs FROM pg_proc 
WHERE proname LIKE 'neuran%' 
ORDER BY proname;

-- Test manual worker execution (should not crash)
DO $$
BEGIN
  PERFORM neuranq_run_once();
  RAISE NOTICE 'neuranq_run_once executed';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'neuranq_run_once not available (optional)';
END$$;
DO $$
BEGIN
  PERFORM neuronmon_sample();
  RAISE NOTICE 'neuronmon_sample executed';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'neuronmon_sample not available (optional)';
END$$;


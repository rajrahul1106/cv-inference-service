import { useCallback, useEffect, useState } from "react";

import { listJobs, submitJob, submitJobFile } from "../api/client.js";
import JobDetail from "../components/JobDetail.jsx";
import JobList from "../components/JobList.jsx";
import UploadCard from "../components/UploadCard.jsx";

// WebSockets handle the selected job's live status; this poll keeps the whole
// list fresh (jobs created in other tabs/sessions have no socket here).
const POLL_INTERVAL_MS = 5000;

export default function Dashboard() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [listError, setListError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listJobs({ limit: 50 });
      setJobs(data.jobs);
      setListError(null);
    } catch (err) {
      setListError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  // Called by UploadCard; lets errors propagate so the form can display them.
  const handleSubmit = useCallback(
    async (input_url, model_type, options) => {
      const submitted = await submitJob(input_url, model_type, options);
      await refresh();
      setSelectedJobId(submitted.job_id);
      return submitted;
    },
    [refresh]
  );

  // Same flow as handleSubmit, but for a multipart file upload.
  const handleSubmitFile = useCallback(
    async (file, model_type, options) => {
      const submitted = await submitJobFile(file, model_type, options);
      await refresh();
      setSelectedJobId(submitted.job_id);
      return submitted;
    },
    [refresh]
  );

  const selectedJob = jobs.find((j) => j.job_id === selectedJobId) || null;

  return (
    <div className="space-y-6">
      <UploadCard onSubmit={handleSubmit} onSubmitFile={handleSubmitFile} />

      {listError && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          Couldn&apos;t load jobs: {listError}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(300px,1fr)_2fr]">
        <JobList
          jobs={jobs}
          selectedJobId={selectedJobId}
          onSelect={setSelectedJobId}
        />
        <JobDetail job={selectedJob} onJobUpdated={refresh} />
      </div>
    </div>
  );
}

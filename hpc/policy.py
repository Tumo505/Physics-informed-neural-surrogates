"""Pure scheduling/resource policy, independently testable without a cluster."""
def segment_size(seconds_left,remaining,maximum,per_step_seconds,reserve=600):
 return max(0,min(maximum,remaining,int((seconds_left-reserve)//(per_step_seconds+120))))
def memory_reason(available_bytes,rss_bytes,allocated_gib,reserve_gib):
 if available_bytes<reserve_gib*2**30:return 'host-wide free-memory reserve'
 if rss_bytes>.8*allocated_gib*2**30:return '80 percent of allocated memory'
 return None

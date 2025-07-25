using System.Collections.Generic;
using System.Collections;
using UnityEngine;


public class CheckpointSingle : MonoBehaviour
{
    private TrackCheckPoints trackCheckpoints;

   private void OnTriggerEnter(Collider other)
{
    if (other.TryGetComponent<AGENT>(out AGENT agent))
    {
        trackCheckpoints.PlayerTroughCheckpoint(this, other.transform);
    }
}

    public void SetTrackCheckpoints(TrackCheckPoints trackCheckpoints)
    {
        this.trackCheckpoints = trackCheckpoints;
    }
}
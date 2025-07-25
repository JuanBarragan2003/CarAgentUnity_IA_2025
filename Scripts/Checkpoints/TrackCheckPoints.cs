using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrackCheckPoints : MonoBehaviour
{
    private List<CheckpointSingle> checkpointSingleList;
    private int nextCheckpointSingleIndex;
    public event System.EventHandler<CarCorrectCheckpointEventArgs> OnCarCorrectCheckpoint;
    public event System.EventHandler<CarWrongCheckpointEventArgs> OnCarWrongCheckpoint;

    private void Awake()

    {
        Transform checkpointsTransform = transform.Find("Checkpoints");
        checkpointSingleList = new List<CheckpointSingle>();
        foreach (Transform checkpointSingleTransform in checkpointsTransform)
        {
            CheckpointSingle checkpointSingle = checkpointSingleTransform.GetComponent<CheckpointSingle>();
            checkpointSingle.SetTrackCheckpoints(this);
            checkpointSingleList.Add(checkpointSingle);
        }
        nextCheckpointSingleIndex = 0;
    }
    public void PlayerTroughCheckpoint(CheckpointSingle checkpointSingle, Transform carTransform)
    {
        if (checkpointSingleList.IndexOf(checkpointSingle) == nextCheckpointSingleIndex)
        {
            Debug.Log("bien");
            nextCheckpointSingleIndex = (nextCheckpointSingleIndex + 1) % checkpointSingleList.Count;

            // Emitir evento correcto
            if (OnCarCorrectCheckpoint != null)
            {
                OnCarCorrectCheckpoint(this, new CarCorrectCheckpointEventArgs { carTransform = carTransform });
            }
        }
        else
        {
            Debug.Log("mal");

            // Emitir evento incorrecto
            if (OnCarWrongCheckpoint != null)
            {
                OnCarWrongCheckpoint(this, new CarWrongCheckpointEventArgs { carTransform = carTransform });
            }
        }
    }
    public void ResetCheckpoint(Transform carTransform)
    {
    nextCheckpointSingleIndex = 0;
    }

    public CheckpointSingle GetNextCheckpointForward(Transform carTransform)
    {
    return checkpointSingleList[nextCheckpointSingleIndex];
    }

}
public class CarCorrectCheckpointEventArgs : System.EventArgs
{
    public Transform carTransform;
}

public class CarWrongCheckpointEventArgs : System.EventArgs
{
    public Transform carTransform;
}
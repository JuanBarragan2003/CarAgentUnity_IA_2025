using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using TMPro;
using UnityEditor.Build.Reporting;
using InputsManager = UnityEngine.InputSystem.InputSystem;

public class AGENT : Agent
{
    [SerializeField] private TrackCheckPoints trackCheckPoints;
    [SerializeField] private Transform spawnPosition;
    [SerializeField] private TMP_Text debugText;

   [SerializeField] private SimpleCarController player;

    private Rigidbody agentRb;

    private int stepCounter = 0;
    private float timeCounter = 0f;
    private int idleSteps = 0;
    private const float idleSpeedThreshold = 0.1f;
    private const int maxIdleSteps = 200;

    private new void Awake()
    {
        agentRb = GetComponent<Rigidbody>();
        if (agentRb == null)
        {
            Debug.LogError("Rigidbody no encontrado en el agente.");
        }
        player = GetComponent<SimpleCarController>();
        if (player == null)
        {
            Debug.LogError("SimpleCarController no encontrado en el agente.");
        }
    }

    private void Start()
    {
        trackCheckPoints.OnCarCorrectCheckpoint += trackCheckPoints_OnCarCorrectCheckpoint;
        trackCheckPoints.OnCarWrongCheckpoint += trackCheckPoints_OnCarWrongCheckpoint;
    }

    private void trackCheckPoints_OnCarWrongCheckpoint(object sender, CarWrongCheckpointEventArgs e)
    {
        if (e.carTransform == transform)
        {
            AddReward(-1f);
        }
    }

    private void trackCheckPoints_OnCarCorrectCheckpoint(object sender, CarCorrectCheckpointEventArgs e)
    {
        if (e.carTransform == transform)
        {
            AddReward(1f);
        }
    }

    public override void OnEpisodeBegin()
    {
        // Reposiciona al agente
        transform.localPosition = spawnPosition.localPosition + new Vector3(Random.Range(-2f, 2f), 0, Random.Range(-2f, 2f));

        // Asegura que mire hacia adelante correctamente
        transform.rotation = spawnPosition.rotation;

        // Reinicia la velocidad del Rigidbody completamente
        agentRb.linearVelocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;

        // reinicar la recompensa acumulada
        SetReward(0f);
        // Reinicia variables de entrenamiento
        trackCheckPoints.ResetCheckpoint(transform);
        player.StopCompletely();  // Por si tienes lógica adicional aquí
        stepCounter = 0;
        timeCounter = 0f;
        idleSteps = 0;
        if (debugText != null)
        {
            debugText.text = "Episodio iniciado. ¡Buena suerte!";
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 checkpointForward = trackCheckPoints.GetNextCheckpointForward(transform).transform.forward;
        float directionDot = Vector3.Dot(transform.forward, checkpointForward);
        sensor.AddObservation(directionDot); // 1 observación CHECHPOINT
        sensor.AddObservation(agentRb.linearVelocity.magnitude); // 2da observación (velocidad)
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        stepCounter++;
        timeCounter += Time.deltaTime;

        float throttle = 0f;
        float steering = 0f;

        float speed = agentRb.linearVelocity.magnitude;


        // 🟥 Penalización por quedarse quieto
        if (speed < idleSpeedThreshold)
        {
            idleSteps++;
            if (idleSteps >= maxIdleSteps)
            {
                AddReward(-0.2f);
                Debug.Log("⚠️ Penalización por estar quieto demasiado tiempo.");
                idleSteps = 0;
            }
        }
        else
        {
            idleSteps = 0;
        }

        Debug.Log($"Pasos: {stepCounter}, Tiempo: {timeCounter:F1}s, Velocidad: {speed:F2}, Recompensa acumulada: {GetCumulativeReward():F4}");

        switch (actions.DiscreteActions[0])
        {
            case 0: throttle = 0f; break;
            case 1: throttle = 1f; break;
            case 2: throttle = -1f; break;
        }

        switch (actions.DiscreteActions[1])
        {
            case 0: steering = 0f; break;
            case 1: steering = 1f; break;
            case 2: steering = -1f; break;
        }

        Debug.Log($"Acciones recibidas: Avance={throttle}, Giro={steering}");
        player.SetInputs(throttle, steering);


        //InputsManager.SetAxis("Throttle", throttle);
        //InputsManager.SetAxis("Steering", steering);

        if (debugText != null)
        {
            debugText.text = $"Pasos: {stepCounter}, Tiempo: {timeCounter:F1}s\n" +
                         $"Velocidad: {speed:F2}, Recompensa: {GetCumulativeReward():F4}";
        }
        //  Finalizar si se supera el tiempo máximo permitido
        if (timeCounter >= 90f)
        {
            Debug.Log("⏱️ Tiempo límite alcanzado. Episodio terminado.");
            EndEpisode();
        }

        if (GetCumulativeReward() <= -2500000000000000000000000f)
        {
            Debug.Log("❌ Recompensa muy baja. Episodio terminado.");
            EndEpisode();
        }
        if (GetCumulativeReward() >= 120f)
        {
            Debug.Log("Meta alcanzada. Episodio terminado.");
            EndEpisode();
        }
        // Protección contra valores inválidos
       if (float.IsNaN(transform.position.x) || float.IsInfinity(transform.position.y) ||
           float.IsNaN(agentRb.linearVelocity.magnitude) || float.IsInfinity(agentRb.linearVelocity.magnitude))
       {
     Debug.LogWarning("⚠️ Valores NaN o Infinity detectados. Terminando episodio.");
     EndEpisode();
    }
    }


   public override void Heuristic(in ActionBuffers actionsOut)
    {
        int throttle = 0; // 0: Quieto, 1: Adelante, 2: Atrás
       int steering = 0; // 0: Sin giro, 1: Derecha, 2: Izquierda

        // Movimiento hacia adelante o atrás
        if (Input.GetKey(KeyCode.W))
            throttle = 1; // Adelante
        else if (Input.GetKey(KeyCode.S))
            throttle = 2; // Atrás
        else
           throttle = 0; // Quieto

        // Giro izquierda o derecha
        if (Input.GetKey(KeyCode.A))
            steering = 2; // Izquierda
       else if (Input.GetKey(KeyCode.D))
           steering = 1; // Derecha
        else
            steering = 0; // Sin giro

    ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = throttle; // Avance
        discreteActions[1] = steering; // Giro
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            Debug.Log("💥 Colisión con la pared (por tag): " + collision.gameObject.name);
            AddReward(-0.5f);
        }

    }
    
    // En tu script del agente, añade esto para debug
    public override void Initialize()
    {

        // Método alternativo para acceder a BehaviorParameters
        var behaviorParams = this.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        Debug.Log($"Behavior Name: {behaviorParams.BehaviorName}");
        Debug.Log($"Model loaded: {behaviorParams.Model != null}");
        Debug.Log($"Behavior Type: {behaviorParams.BehaviorType}");
    }
    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            float contactForce = collision.relativeVelocity.magnitude;

            if (contactForce > 2f)
            {
                Debug.Log("🧱 Pegado fuerte a la pared");
                AddReward(-0.2f);
            }
            else
            {
                Debug.Log("🚧 Rozando ligeramente la pared");
                AddReward(-0.05f);
            }
        }
    }

} 
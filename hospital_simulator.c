
// hospital_simulator.c
#include <stdio.h>
#include<stdlib.h>
#include<float.h>
#include<stdbool.h>
#include <math.h>

// #include "globals.h"
// #include "event_queue.h"
// #include "priority_queue.h"


//for event_type:
// 0 represents arrival to first queue
// 1 represents patient entering second queue
// 2 represents patient leaving second queue
// 3 represent janitor finishing
typedef struct event {
  double time;
  short event_type;
} event;

event* new_event(double time, int event_type) {
  event* ret = (event*)malloc(sizeof(event));
  ret->time = time;
  ret->event_type = event_type;
  return ret;
}

typedef struct MinHeap {
    event** array;     // array of pointers to event pointers
    int capacity;
    int size;
} MinHeap;

double lambda;
double mu_E;
double mu_T;
double mu_C;
int B;
int R;
int m1;
int m2;
int S;
int total_nodes;
int departures = 0;
int dropped_count;
double current_time = 0.0;
double last_event_time = 0.0;
int nodes_in_system = 0;
MinHeap* heap;
int began_evalu = 0;
int being_served = 0;
int available_rooms;
int janitors_working = 0;
double roomCleanupTime = 0.0;
int rooms_to_clean = 0;
int began_treatment = 0;
double cumulative_area = 0.0;
int rooms_cleaned = 0;
double everyone_arrival_time = 0.0;
double everyone_departu_time = 0.0;

void swap(void** a, void** b) {
    void* temp = *a;
    *a = *b;
    *b = temp;
}

MinHeap* createMinHeap(int capacity) {
    MinHeap* Minheap = (MinHeap*)malloc(sizeof(MinHeap));
    Minheap->capacity = capacity;
    Minheap->size = 0;
    Minheap->array = (event**)malloc(capacity * sizeof(event*)); // Allocate memory for array of event pointers;
    return Minheap;
}

void resize(MinHeap* minHeap, int new_capacity) {
    minHeap->array = (event**)realloc(minHeap->array, new_capacity * sizeof(event*));
    if (minHeap->array == NULL) {
        printf("Memory reallocation failed\n");
        return;
    }
    minHeap->capacity = new_capacity;
}

void minHeapify(MinHeap* minHeap, int index) {
    int smallest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    // Check if left child exists and is smaller than current smallest,
    // or if left child has the same time but smaller event_type
    if (left < minHeap->size &&
        (minHeap->array[left]->time < minHeap->array[smallest]->time ||
         (minHeap->array[left]->time == minHeap->array[smallest]->time &&
          minHeap->array[left]->event_type < minHeap->array[smallest]->event_type)))
        smallest = left;

    // Check if right child exists and is smaller than current smallest,
    // or if right child has the same time but smaller event_type
    if (right < minHeap->size &&
        (minHeap->array[right]->time < minHeap->array[smallest]->time ||
         (minHeap->array[right]->time == minHeap->array[smallest]->time &&
          minHeap->array[right]->event_type < minHeap->array[smallest]->event_type)))
        smallest = right;

    // If the smallest element is not the current index, swap it and heapify recursively
    if (smallest != index) {
        swap((void**)&minHeap->array[index], (void**)&minHeap->array[smallest]);
        minHeapify(minHeap, smallest);
    }
}

void insertEvent(MinHeap* minHeap, event* newEvent) {
    if (minHeap->size == minHeap->capacity) {
        // If array is full, double its capacity
        resize(minHeap, minHeap->capacity * 2);
    }

    int i = minHeap->size;
    minHeap->array[i] = newEvent; // Store the pointer to the event
    minHeap->size++;

    // Heapify the new element to maintain heap property
    while (i != 0 && minHeap->array[(i - 1) / 2]->time > minHeap->array[i]->time) {
        swap((void**)&minHeap->array[i], (void**)&minHeap->array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

event* extractMin(MinHeap* minHeap) {
    if (minHeap->size <= 0) {
        printf("Underflow: Could not extract minimum element\n");
        return NULL;
    }

    if (minHeap->size == 1) {
        minHeap->size--;
        return minHeap->array[0];
    }

    event* root = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    minHeap->size--;

    // If size is less than one-half of capacity, halve the capacity
    if (minHeap->size < minHeap->capacity / 2) {
        resize(minHeap, minHeap->capacity / 2);
    }

    minHeapify(minHeap, 0);
    return root;
}

void free_heap(MinHeap* minHeap) {
    // Free memory allocated for each event pointer in the array
    for (int i = 0; i < minHeap->size; i++) {
        free(minHeap->array[i]); // Free memory for the event structure
    }

    // Free memory allocated for the array of event pointers
    free(minHeap->array);

    // Free memory allocated for the MinHeap structure itself
    free(minHeap);
}

void print_heap(MinHeap* MinHeap) {
  for (int i = 0; i < MinHeap->capacity; i++) {
    printf("%f  %i\n", MinHeap->array[i]->time, MinHeap->array[i]->event_type);
  }
}

typedef struct waitingQueueNode{
    double arrival_time;
    double evalu_time;
    double priority;
    struct waitingQueueNode* next;
} waitingQueueNode;

typedef struct PriorityQueueNode{
    double arrival_time;
    double time_moved_to_next_queue;
    double treatm_time;
    double priority;
    double departure_time;
} PriorityQueueNode;

typedef struct janitors{
    double* departure_times;
    int capacity;
    int size;
} janitors;

typedef struct PriorityQueue{
    PriorityQueueNode** heapArray;
    PriorityQueueNode* top; //similar to the head of the regular queue
    int capacity;
    int size;
    int waiting_count;
    double cumulative_waiting;
    bool idle;
} PriorityQueue;

PriorityQueue* pq;

typedef struct waitingQueue{
    waitingQueueNode* head;
    waitingQueueNode* tail;

    waitingQueueNode* first;  // Point to the first arrived customer that is waiting for service
    waitingQueueNode* last;   // Point to the last arrrived customer that is waiting for service
    int waiting_count;     // Current number of customers waiting for service

    double cumulative_waiting;  // Accumulated waiting time for all effective departures
    double cumulative_area;   // Accumulated number of customers in the system multiplied by their residence time, for E[n] computation

    bool idle; //indicates whether the system is idle or not. Added by I.
} waitingQueue;

void freeQueue(waitingQueue* q) {
    waitingQueueNode* curr = q->head;
    waitingQueueNode* temp;
    while (curr) {
      temp = curr;
      curr = curr->next;
      free(temp);  
    }
    free(q);
}

waitingQueue* elementQ;

PriorityQueue* createPriorityQueue(int capacity) {
    PriorityQueue* pq = (PriorityQueue*)malloc(sizeof(PriorityQueue));
    pq->heapArray = (PriorityQueueNode**)malloc(capacity * sizeof(PriorityQueueNode*));
    pq->capacity = capacity;
    pq->size = 0;
    pq->idle = 1;
    pq->waiting_count = 0;
    pq->cumulative_waiting = 0;
    pq->top = NULL;
    return pq;
}

janitors* createJanitors(int capacity) {
    janitors* jn = (janitors*)malloc(sizeof(janitors));
    jn->departure_times = (double*)malloc(sizeof(double) * capacity);
    for (int i = 0; i < capacity; i++) jn->departure_times[i] = 0.0;
    jn->capacity = capacity;
    jn->size = 0;
    return jn;
}

janitors* jn;

void freeJanitors(janitors* jn) {
    free(jn->departure_times);
    free(jn);
}

void add_departure_time(janitors* jn, double departure_time) {
    if (jn->size >= jn->capacity) {
        printf("Full array\n");
        return;
    }
    jn->departure_times[jn->size] = departure_time;
    jn->size++;
}

//helper function that removes the head and returns it
waitingQueueNode* extract(waitingQueue* q) {
  if (!q || !q->head) return NULL;

  waitingQueueNode* ret = q->head;
  waitingQueueNode* new_head = ret->next;

  q->head = new_head;
  return ret;
}

PriorityQueueNode* getNext(PriorityQueue* pq, PriorityQueueNode* first) {
    if (pq->size == 0 || !first) {
        printf("Priority Queue is empty or provided pointer is invalid\n");
        return NULL;
    }
    
    int i;
    for (i = 0; i < pq->size; ++i) {
        if (pq->heapArray[i] == first) {
            // If the current element is found, return the next element
            if (i + 1 < pq->size) {
                return pq->heapArray[i + 1];
            } else {
                // If the current element is the last element in the queue
                printf("No next element\n");
                return NULL;
            }
        }
    }
    
    // If the provided pointer is not found in the priority queue
    printf("Provided pointer does not belong to the priority queue\n");
    return NULL;
}

void heapifyUp(PriorityQueue* pq, int index) {
    while (index > 0 && pq->heapArray[index]->priority < pq->heapArray[(index - 1) / 2]->priority) {
        swap((void**)&pq->heapArray[index], (void**)&pq->heapArray[(index - 1) / 2]);
        index = (index - 1) / 2;
    }
}

void heapifyDown(PriorityQueue* pq, int index) {
    int smallest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < pq->size && pq->heapArray[left]->priority < pq->heapArray[smallest]->priority) {
        smallest = left;
    }

    if (right < pq->size && pq->heapArray[right]->priority < pq->heapArray[smallest]->priority) {
        smallest = right;
    }

    if (smallest != index) {
        swap((void**)&pq->heapArray[index], (void**)&pq->heapArray[smallest]);
        heapifyDown(pq, smallest);
    }
}

void enqueue(PriorityQueue* pq, PriorityQueueNode* newNode) {
    if (pq->size == pq->capacity) {
        printf("Priority Queue is full\n");
        return;
    }
    pq->heapArray[pq->size] = newNode;
    pq->size++;
    heapifyUp(pq, pq->size - 1);

    // Update pointer to the top element
    if (pq->size == 1 || newNode->priority < pq->top->priority) {
        pq->top = newNode;
    }
}

PriorityQueueNode* dequeue(PriorityQueue* pq) {
    if (pq->size == 0) {
        printf("Priority Queue is empty (dequeue)\n");
        return NULL;
    }
    PriorityQueueNode* dequeuedNode = pq->heapArray[0];
    pq->heapArray[0] = pq->heapArray[pq->size - 1];
    pq->size--;
    heapifyDown(pq, 0);

    // Update pointer to the first element
    if (pq->size == 0) {
        pq->top = NULL;
    } else {
        pq->top = pq->heapArray[0];
    }

    return dequeuedNode;
}

void freePriorityQueue(PriorityQueue* pq) {
    while (pq->top) {
        PriorityQueueNode* node = dequeue(pq); // Dequeue the next node
        free(node); // Free the dequeued node
    }
    free(pq->heapArray); // Free the array of pointers
    free(pq); // Free the priority queue structure itself
}

PriorityQueueNode* peek(PriorityQueue* pq) {
    if (pq->size == 0) {
        printf("Priority Queue is empty\n");
        return NULL;
    }
    return pq->heapArray[0];
}

waitingQueueNode* create_node(double arrival_time, double evalu_time) {
  waitingQueueNode* ret = (waitingQueueNode*)malloc(sizeof(waitingQueueNode));
  ret->arrival_time = arrival_time;
  ret->evalu_time = evalu_time;
  ret->next = NULL;
  return ret;
}

PriorityQueueNode* create_pq_node(
    double arrival_time, double next_queue_time, double treatment_time, double priority
) {
    PriorityQueueNode* ret = (PriorityQueueNode*)malloc(sizeof(PriorityQueueNode));
    ret->arrival_time = arrival_time;
    ret->treatm_time = treatment_time;
    ret->time_moved_to_next_queue = next_queue_time;
    ret->priority = priority;
    ret->departure_time = 0;
    return ret;
}

void insert(waitingQueue* q, waitingQueueNode* new_tail) {
  if (!q) return; //no queue
  if (!q->head) { //empty queue
    q->head = new_tail;
    q->tail = new_tail;
  }
  else {
    q->tail->next = new_tail;
    q->tail = new_tail;
  }
}

double generateExponential(double lambda) {
    double u = rand() / (RAND_MAX + 1.0);  // Generate a random number between 0 and 1
    return -log(1 - u) / lambda;  // Inverse transform for exponential distribution
}

double generate_uniform(double a, double b) {
    return a + (rand() / (RAND_MAX / (b - a)));
}

double generateTreatmentTime(double mu, double priority) {
    if (priority > 0.5) return generateExponential(mu);
    else return generateExponential(2.0 * mu);
}

waitingQueue* InitializeQueue(){
  waitingQueue* ret = (waitingQueue*)malloc(sizeof(waitingQueue));

  total_nodes = 0;
  //initialize variables
  ret->head = NULL;
  ret->tail = NULL;
  ret->first = NULL;
  ret->last = NULL;
  ret->waiting_count = 0;
  ret->cumulative_waiting = 0;
  ret->cumulative_area = 0.0;
  ret->idle = true;

  srand(S); //seed the random number generator

  double arrival_time = 0.0;

  while (arrival_time < 1800.0) { //until 6AM
    arrival_time += generateExponential(lambda);
    waitingQueueNode* elt = create_node(arrival_time, 0);
    insert(ret, elt);
    total_nodes++;
  }

  //generate service times
  waitingQueueNode* node = ret->head;
  while (node) { //until end of linked list
    double evalu_time = generateExponential(mu_E);
    node->evalu_time = evalu_time;
    node = node->next;
  }

  pq = createPriorityQueue(total_nodes);
  heap = createMinHeap(total_nodes);
  jn = createJanitors(total_nodes);

  node = ret->head;

  struct event* head_arrival = new_event(node->arrival_time, 0);
  insertEvent(heap, head_arrival); //insert the head's arrival

  return ret;
}

waitingQueueNode* dropNode(waitingQueueNode* arrival) {
    if (!elementQ || !elementQ->head || !arrival) return NULL;

    if (current_time >= 360.0)
        dropped_count++;

    nodes_in_system--;

    waitingQueueNode* current = elementQ->head;
    waitingQueueNode* prev = NULL;

    waitingQueueNode* ret = arrival->next;

    // Traverse the queue to find the node to drop
    while (current != arrival && current != NULL) {
        prev = current;
        current = current->next;
    }

    // If the node was found
    if (current == arrival) {
        // Adjust pointers to bypass the node to drop
        if (prev == NULL) {
            // If the node to drop is the head of the queue
            elementQ->head = current->next;
        } else {
            // If the node to drop is in the middle or end of the queue
            prev->next = current->next;
        }
        free(current); // Free memory allocated for the dropped node
    }

    return ret; // Return the next element of the queue
}

void StartEvalu() {
    if (!elementQ || !elementQ->head || !elementQ->first){
        return;
    } 

    if (current_time >= 360.0) {
        elementQ->cumulative_waiting += (current_time - elementQ->first->arrival_time);
        began_evalu++;
    }

    elementQ->waiting_count--;
    
    being_served++;
    if (being_served == m1) elementQ->idle = false;

    //schedule entry into next queue
    double next_queue_time = current_time + elementQ->first->evalu_time;
    insertEvent(heap, new_event(next_queue_time, 1));

    if (elementQ->waiting_count == 0) { //only one element that is waiting for service
        elementQ->first= NULL;
        elementQ->last = NULL;
    }
    else {
        elementQ->first = elementQ->first->next;
    }
  
}

waitingQueueNode* begin_arrival(waitingQueueNode* arrival) {
    if (!elementQ || !elementQ->head || !arrival) return NULL;

    waitingQueueNode* next_node;
    nodes_in_system++;
    if (nodes_in_system > B) { //capacity reached
        next_node = dropNode(arrival);

        if (next_node) {
            insertEvent(heap, new_event(next_node->arrival_time, 0));
        }

        return next_node;
    }

    if (!elementQ->first) {
        elementQ->first = arrival;
    }
    elementQ->last = arrival;

    next_node = arrival->next;

    if (next_node) { //process next node's arrival
        insertEvent(heap, new_event(next_node->arrival_time, 0));
    }

    elementQ->waiting_count++;
    if (being_served < m1) { //system has open nurses
        StartEvalu();
    }

    return next_node;
}

void BeginTreatment() {
    if (!pq || !pq->top) return;

    available_rooms--;
    pq->waiting_count--;

    PriorityQueueNode* node = dequeue(pq);
    double departure_time = current_time + node->treatm_time;

    if (current_time >= 360.0) {
        pq->cumulative_waiting += (current_time - node->time_moved_to_next_queue);
        began_treatment++;
        if (departure_time <= 1800) //if node departs within the 30 hours
            everyone_arrival_time += node->arrival_time;
            //everyone_arrival_time is used to calculate the average response time
    }

    //schedule node's departure
    insertEvent(heap, new_event(departure_time, 2));

    free(node);
    return;
}

void NextQueue() {
    waitingQueueNode* node = extract(elementQ);
    if (!node) {
        printf("something is wrong\n");
        return;
    }
    double priority = generate_uniform(0, 1);
    double treatment_time = generateTreatmentTime(mu_T, priority);
    PriorityQueueNode* pNode = create_pq_node(
        node->arrival_time, current_time, treatment_time, priority
    );
    being_served--;

    free(node);
    
    enqueue(pq, pNode); // add to priority queue
    pq->waiting_count++;

    if (available_rooms > 0) { // there is an available room for someone
        BeginTreatment();
    }

    if (elementQ->waiting_count > 0) { // another node can start evaluation
        StartEvalu();
    }
    return;
}

void clean_room() { //janitor starts cleaning room
    janitors_working++;
    rooms_to_clean--;
    double janitor_finishes = current_time + (1 / mu_C);
    //schedule event for when the janitor finishes
    insertEvent(heap, new_event(janitor_finishes, 3));
}

void NodeDeparture() {
    if (!pq) return;

    if (current_time >= 360.0) {
        departures++;
        everyone_departu_time += current_time;
    }
    nodes_in_system--;
    rooms_to_clean++;
    add_departure_time(jn, current_time); // add to array of departure times
    // used for janitor times

    if (janitors_working < m2) { //there is an available janitor
        clean_room();
    }
}

void printJanitors(janitors* jn) {
    for (int i = 0; i < jn->size; i++) {
        printf("%lf\n", jn->departure_times[i]);
    }
}

void janitor_done() {
    available_rooms++;
    janitors_working--;
    double cleanup = current_time - jn->departure_times[rooms_cleaned];
    roomCleanupTime += cleanup;
    rooms_cleaned++;
    if (pq->waiting_count > 0) { //there is a node that is waiting for treatment
        BeginTreatment();
    }
    if (rooms_to_clean > 0) { // there is a room that needs to be claned
        clean_room();
    }
}

void printStatistics() { //avoid first six hours
    printf("Departures = %i\n", departures);
    printf("Mean_num_patients = %lf\n", cumulative_area / (current_time - 360.0));
    if (departures == 0) {
        printf("Mean_response_time = No departures\n");
    }
    else
        printf("Mean_response_time = %lf\n", (everyone_departu_time - everyone_arrival_time) / (double) departures);
    if (began_evalu == 0)
        printf("Mean_wait_E_queue = no patients have responded\n");
    else
        printf("Mean_wait_E_queue = %lf\n", elementQ->cumulative_waiting / (double) began_evalu);
    if (began_treatment == 0)
        printf("Mean_wait_P_queue = no patients began evaluation\n");
    else
        printf("Mean_wait_P_queue = %lf\n", pq->cumulative_waiting / (double) began_treatment);
    if (rooms_cleaned == 0)
        printf("Mean_cleanup_time = no rooms got cleaned\n");
    else
        printf("Mean_cleanup_time = %lf\n", roomCleanupTime / (double) rooms_cleaned);
    printf("Dropped_arrivals = %i\n", dropped_count);
}

void printZeroStatistics() {
    printf("Departures = 0\n");
    printf("Mean_num_patients = 0\n");
    printf("Mean_response_time = 0\n");
    printf("Mean_wait_E_queue = 0\n");
    printf("Mean_wait_P_queue = 0\n");
    printf("Mean_cleanup_time = 0\n");
    printf("Dropped_arrivals = 0\n");
}

void simulation() {

  waitingQueueNode* node = elementQ->head;
  //int total_departures = departures + dropped_count;
  while (current_time < 1800.0) {
    event* min_event = extractMin(heap);

    if (!min_event) {
      free(min_event);
      return;
    }

    current_time = min_event->time;
    double time_passed = current_time - last_event_time;
    if (current_time >= 360.0) {
        cumulative_area += (time_passed * nodes_in_system);
    }
    last_event_time = current_time;
    short type_of_event = min_event->event_type;
    free(min_event);

    if (type_of_event == 0) { //arrival for waiting area
      //call function for waiting area arrival
      node = begin_arrival(node);
    }
    else if (type_of_event == 1) { //patient enters second queue
        NextQueue();
    }
    else if (type_of_event == 2) { //patient leaves second queue
        NodeDeparture();
    }
    else if (type_of_event == 3) { //janitor finishes
        janitor_done();
    }
  }

  printStatistics();
  free_heap(heap);
  freeQueue(elementQ);
  freePriorityQueue(pq);
  freeJanitors(jn);
}

///////////////////////////
// performance metrics to keep track of
// Current number of patients in emergency department
// Response time for each departing patient (time between when a patient enters and leaves the whole system)
// waiting time in E Queue
// Waiting time in the P queue for each patient
// Waiting time for cleanup for each room
// Number of patients who leave the system because it’s at full capacity when they arrive
/////////////////////////////////////

int main(int argc, char* argv[]) {
    if (argc < 10) {
        printf("Too few input arguments\n");
        return 1;
    }

    lambda = atof(argv[1]);
    mu_E = atof(argv[2]);
    mu_T = atof(argv[3]);
    mu_C = atof(argv[4]);
    B = atoi(argv[5]);
    R = atoi(argv[6]);
    available_rooms = R;
    m1 = atoi(argv[7]);
    m2 = atoi(argv[8]);
    S = atoi(argv[9]);

    if (lambda <= 0) {
        printf("lambda is non positive\n");
        printZeroStatistics();
        return 2;
    }

    if (
        mu_E <= 0 || mu_T <= 0 || mu_C <= 0 || B <= 0 || R <= 0 || m1 <= 0 || m2 <= 0 || S <= 0
    ) {
        printf("Input must be positive\n");
        return 3;
    }

    elementQ = InitializeQueue();
    simulation();
    return 0;
}
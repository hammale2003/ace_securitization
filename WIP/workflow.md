# WORKFLOW DU SYSTÈME ACE (Agentic Context Engineering)

## Vue d'ensemble

Le système ACE est un framework d'apprentissage automatique qui utilise trois agents spécialisés pour traiter des questions et faire évoluer une base de connaissances (playbook) au fil du temps. Le système est conçu pour le domaine de la titrisation (securitization) et de la finance structurée.

---

## ARCHITECTURE GLOBALE

Le système fonctionne selon un pipeline séquentiel :

```
UTILISATEUR (Question)
    ↓
┌─────────────────────────────────┐
│  AGENT 1: GENERATOR            │
│  Génère une réponse             │
└─────────────────────────────────┘
    ↓ (GeneratorOutput)
┌─────────────────────────────────┐
│  AGENT 2: REFLECTOR            │
│  Analyse et identifie erreurs   │
└─────────────────────────────────┘
    ↓ (ReflectorOutput)
┌─────────────────────────────────┐
│  AGENT 3: CURATOR               │
│  Met à jour le playbook         │
└─────────────────────────────────┘
    ↓ (CuratorOutput)
┌─────────────────────────────────┐
│  PLAYBOOK MANAGER               │
│  Applique les modifications     │
└─────────────────────────────────┘
    ↓
PLAYBOOK ENRICHI + RÉSULTAT FINAL
```

---

## ORCHESTRATION DES AGENTS

### Vue d'ensemble de l'orchestration

L'orchestration des agents dans le système ACE est gérée par la classe **`ACEPipeline`**, qui agit comme un **coordinateur central**. Cette classe est responsable de :

1. **Initialiser** tous les agents et leurs dépendances
2. **Séquencer** l'exécution des agents dans le bon ordre
3. **Coordonner** le passage de données entre les agents
4. **Gérer** les ressources partagées (client LLM, playbook manager)
5. **Orchestrer** les callbacks de streaming pour chaque agent

### Architecture d'orchestration

```
┌────────────────────────────────────────────────────────────┐
│                    ACEPipeline                             │
│  (Orchestrateur Principal)                                 │
│                                                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Ressources Partagées                              │    │
│  │  - LLMClient (client unique pour tous les agents)  │    │
│  │  - PlaybookManager (gestion du playbook)           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Generator   │  │  Reflector   │  │   Curator    │      │
│  │  (Agent 1)   │  │  (Agent 2)   │  │  (Agent 3)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│       │                  │                  │              │
│       └──────────────────┴──────────────────┘              │
│                    (Séquence d'exécution)                  │
└────────────────────────────────────────────────────────────┘
```

### Initialisation des agents

Lors de la création d'une instance `ACEPipeline`, tous les agents sont initialisés **une seule fois** :

```python
def __init__(self, config: ACEConfig = None):
    self.config = config or ACEConfig()
    
    # 1. Initialisation du client LLM (partagé par tous les agents)
    self.client = create_client(self.config.llm)
    
    # 2. Initialisation du PlaybookManager (gestionnaire de playbook)
    self.playbook_manager = PlaybookManager(self.config.playbook)
    
    # 3. Initialisation des agents (chacun reçoit les ressources nécessaires)
    self.generator = Generator(self.client)  # Reçoit le client LLM
    self.reflector = Reflector(self.client, self.config.max_reflector_iterations)  # Reçoit le client + config
    self.curator = Curator(self.client, self.playbook_manager)  # Reçoit le client + le manager
```

**Points clés** :
- **Client LLM unique** : Tous les agents partagent le même `LLMClient`, garantissant la cohérence des appels API
- **PlaybookManager partagé** : Seul le Curator a une référence directe au manager, mais tous les agents accèdent au playbook via le pipeline
- **Configuration centralisée** : La configuration est stockée dans `ACEPipeline` et distribuée aux agents selon leurs besoins

### Flux d'orchestration séquentiel

L'orchestration suit un **pattern séquentiel strict** où chaque agent attend la fin de l'agent précédent :

```python
def run(self, question, ground_truth=None, feedback=None, stream_callbacks=None):
    # ÉTAPE 0 : Préparation
    callbacks = stream_callbacks or {}
    playbook = self.playbook_manager.get_playbook()  # Charge le playbook actuel
    
    # ÉTAPE 1 : GENERATOR (Agent 1)
    generator_output = self.generator.generate(
        question=question,
        playbook=playbook,  # Playbook passé en paramètre
        stream_callback=callbacks.get("generator")  # Callback optionnel
    )
    # ⏸️ Attente de la fin du Generator avant de continuer
    
    # ÉTAPE 2 : REFLECTOR (Agent 2)
    reflector_output = self.reflector.reflect(
        question=question,
        generator_output=generator_output,  # ⬅️ Utilise la sortie du Generator
        playbook=playbook,  # Même playbook (pas encore mis à jour)
        ground_truth=ground_truth,  # Paramètres additionnels
        feedback=feedback,
        stream_callback=callbacks.get("reflector")
    )
    # ⏸️ Attente de la fin du Reflector avant de continuer
    
    # ÉTAPE 3 : CURATOR (Agent 3)
    curator_output = self.curator.curate(
        question=question,
        generator_output=generator_output,  # ⬅️ Utilise la sortie du Generator
        reflector_output=reflector_output,  # ⬅️ Utilise la sortie du Reflector
        playbook=playbook,  # Même playbook (pas encore mis à jour)
        stream_callback=callbacks.get("curator")
    )
    # ⏸️ Attente de la fin du Curator avant de continuer
    
    # ÉTAPE 4 : Application des mises à jour
    added_bullets = self.curator.apply_updates(
        curator_output,  # ⬅️ Utilise la sortie du Curator
        reflector_output  # ⬅️ Utilise aussi la sortie du Reflector (pour les tags)
    )
    # ⏸️ Les mises à jour sont appliquées au playbook
    
    # ÉTAPE 5 : Récupération des statistiques
    playbook_stats = self.playbook_manager.get_playbook().get_stats()
    
    # ÉTAPE 6 : Retour du résultat complet
    return ACEPipelineResult(
        question=question,
        generator_output=generator_output,
        reflector_output=reflector_output,
        curator_output=curator_output,
        added_bullets=added_bullets,
        playbook_stats=playbook_stats
    )
```

### Passage de données entre agents

Le système utilise un **pattern de chaînage de données** où chaque agent reçoit les sorties des agents précédents :

| Agent | Reçoit | Produit | Utilisé par |
|-------|--------|---------|-------------|
| **Generator** | `question`, `playbook` | `GeneratorOutput` | → Reflector, Curator |
| **Reflector** | `question`, `generator_output`, `playbook`, `ground_truth?`, `feedback?` | `ReflectorOutput` | → Curator |
| **Curator** | `question`, `generator_output`, `reflector_output`, `playbook` | `CuratorOutput` | → `apply_updates()` |

**Important** : Le playbook utilisé par tous les agents dans une même exécution est **le même** (celui chargé au début). Les mises à jour ne sont appliquées qu'à la fin, après que tous les agents aient terminé leur travail.

### Gestion des ressources partagées

#### 1. Client LLM partagé

```python
self.client = create_client(self.config.llm)  # Créé une seule fois

# Tous les agents utilisent le même client
self.generator = Generator(self.client)
self.reflector = Reflector(self.client, ...)
self.curator = Curator(self.client, ...)
```

**Avantages** :
- Cohérence des paramètres (temperature, model, etc.)
- Réutilisation de la connexion API
- Gestion centralisée des erreurs

#### 2. PlaybookManager

```python
self.playbook_manager = PlaybookManager(self.config.playbook)

# Le Curator a une référence directe pour appliquer les mises à jour
self.curator = Curator(self.client, self.playbook_manager)

# Les autres agents reçoivent le playbook via le pipeline
playbook = self.playbook_manager.get_playbook()
```

**Fonctionnalités du PlaybookManager** :
- **Thread-safe** : Utilise un verrou (`RLock`) pour les opérations concurrentes
- **Persistence** : Charge et sauvegarde automatiquement depuis/vers JSON
- **Cache** : Maintient le playbook en mémoire après le premier chargement

### Orchestration du streaming

Le système supporte le **streaming en temps réel** pour chaque agent via des callbacks :

```python
stream_callbacks = {
    "generator": lambda token: print(f"Generator: {token}"),
    "reflector": lambda token: print(f"Reflector: {token}"),
    "curator": lambda token: print(f"Curator: {token}")
}

result = pipeline.run(
    question="...",
    stream_callbacks=stream_callbacks
)
```

**Mécanisme** :
1. Le pipeline reçoit un dictionnaire de callbacks
2. Chaque callback est passé à l'agent correspondant
3. L'agent appelle le callback à chaque token reçu du LLM
4. Le streaming se fait **séquentiellement** : d'abord Generator, puis Reflector, puis Curator

### Gestion des erreurs et interruption

L'orchestration est **synchrone et séquentielle** :
- Si le Generator échoue, le Reflector et le Curator ne sont pas exécutés
- Si le Reflector échoue, le Curator n'est pas exécuté
- Les erreurs remontent au niveau du pipeline

**Pas de gestion d'erreur automatique** : C'est à l'utilisateur du pipeline de gérer les exceptions.

### Modes d'orchestration

Le système offre **3 modes d'orchestration** :

#### 1. Mode complet (`run()`)
```python
result = pipeline.run(question="...")
```
- Exécute les 3 agents en séquence
- Applique les mises à jour au playbook
- Retourne un résultat complet

#### 2. Mode avec raffinement (`run_with_refinement()`)
```python
result = pipeline.run_with_refinement(
    question="...",
    reflector_iterations=5
)
```
- Similaire au mode complet
- Le Reflector effectue plusieurs passes itératives
- Meilleure qualité d'analyse mais plus lent

#### 3. Mode génération seule (`generate_only()`)
```python
answer = pipeline.generate_only(question="...")
```
- Exécute uniquement le Generator
- Pas de réflexion ni de curation
- Utile pour l'inférence après entraînement

### Diagramme de séquence complet

```
Utilisateur
    │
    ├─→ ACEPipeline.run(question, ground_truth?, feedback?)
    │       │
    │       ├─→ PlaybookManager.get_playbook()
    │       │       └─→ Charge playbook.json (si pas en cache)
    │       │
    │       ├─→ Generator.generate(question, playbook)
    │       │       ├─→ LLMClient.chat() ou stream_chat()
    │       │       └─→ Retourne GeneratorOutput
    │       │
    │       ├─→ Reflector.reflect(question, generator_output, playbook, ...)
    │       │       ├─→ LLMClient.chat() ou stream_chat()
    │       │       └─→ Retourne ReflectorOutput
    │       │
    │       ├─→ Curator.curate(question, generator_output, reflector_output, playbook)
    │       │       ├─→ LLMClient.chat() ou stream_chat()
    │       │       └─→ Retourne CuratorOutput
    │       │
    │       ├─→ Curator.apply_updates(curator_output, reflector_output)
    │       │       ├─→ PlaybookManager.update_tags() (via Curator)
    │       │       ├─→ PlaybookManager.apply_operations() (via Curator)
    │       │       └─→ PlaybookManager.save() (automatique)
    │       │
    │       └─→ Retourne ACEPipelineResult
    │
    └─→ Résultat avec toutes les sorties
```

## AGENT 1 : GENERATOR (Générateur)

### Rôle
Le Generator est responsable de produire des réponses aux questions de l'utilisateur en utilisant les connaissances accumulées dans le playbook.

### INPUTS (Entrées)

| Paramètre | Type | Description | Obligatoire |
|-----------|------|-------------|-------------|
| `question` | `str` | La question posée par l'utilisateur | ✅ Oui |
| `playbook` | `Playbook` | L'objet playbook contenant toutes les connaissances (strategies, pitfalls, templates, definitions) | ✅ Oui |
| `stream_callback` | `Callable[[str], None]` | Fonction optionnelle pour recevoir les tokens en streaming en temps réel | ❌ Non |

### Processus interne

1. **Formatage du playbook** : Le playbook est converti en texte formaté pour être inclus dans le prompt
   ```python
   playbook_text = playbook.format_for_prompt()
   ```
   Le format inclut toutes les sections avec leurs bullets, par exemple :
   ```
   ### STRATEGIES
   [str-00001] helpful=0 harmful=0 :: When assessing a true sale...
   [str-00002] helpful=0 harmful=0 :: When structuring credit enhancement...
   
   ### DEFINITIONS
   [def-00001] helpful=0 harmful=0 :: True Sale: A legal term...
   ```

2. **Construction du message** : Le message utilisateur est formaté avec le playbook et la question
   ```python
   user_message = format_generator_user_message(playbook_text, question)
   ```

3. **Appel au LLM** : Le message est envoyé au modèle de langage avec le prompt système du Generator
   - Prompt système : Instructions pour analyser le playbook et générer une réponse JSON
   - Le LLM doit retourner un JSON avec `reasoning`, `bullet_ids`, et `final_answer`

4. **Parsing de la réponse** : La réponse JSON est parsée pour créer un objet `GeneratorOutput`

### OUTPUTS (Sorties) - GeneratorOutput

| Champ | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Le raisonnement étape par étape expliquant comment la réponse a été construite. Indique quels bullets du playbook ont influencé la pensée. |
| `bullet_ids` | `List[str]` | Liste des identifiants des bullets du playbook qui ont été utilisés (ex: `["str-00001", "def-00001"]`) |
| `final_answer` | `str` | La réponse finale, concise et autoritaire, ou la clause rédigée |


### Exemple de sortie

```json
{
    "reasoning": "J'ai analysé la question sur le true sale. J'ai utilisé le bullet str-00001 qui liste les 5 critères essentiels, et le bullet def-00001 qui définit le concept. En appliquant ces critères, je peux conclure que...",
    "bullet_ids": ["str-00001", "def-00001"],
    "final_answer": "Un true sale en titrisation nécessite que : 1) Le transfert d'actifs soit absolu et non comme garantie d'un prêt ; 2) L'originator renonce au contrôle sur les actifs ; 3) Le SPV assume les risques et récompenses de propriété ; 4) La vente soit écrite et enregistrée ; 5) La vente soit à distance et à juste valeur."
}
```

### Cas d'usage

- **Mode streaming** : Si `stream_callback` est fourni et que le client LLM supporte le streaming, les tokens sont envoyés au fur et à mesure
- **Mode normal** : La réponse complète est attendue avant d'être retournée

---

## AGENT 2 : REFLECTOR (Réflecteur)

### Rôle
Le Reflector analyse la réponse du Generator, identifie les erreurs ou points d'amélioration, et extrait des insights qui serviront à améliorer le playbook.

### INPUTS (Entrées)

| Paramètre | Type | Description | Obligatoire |
|-----------|------|-------------|-------------|
| `question` | `str` | La question originale posée par l'utilisateur | ✅ Oui |
| `generator_output` | `GeneratorOutput` | La sortie complète du Generator (reasoning, bullet_ids, final_answer) | ✅ Oui |
| `playbook` | `Playbook` | Le playbook actuel utilisé par le Generator | ✅ Oui |
| `ground_truth` | `Optional[str]` | La réponse correcte attendue (pour comparaison et apprentissage supervisé) | ❌ Non |
| `feedback` | `Optional[str]` | Feedback humain sur la qualité de la réponse | ❌ Non |
| `stream_callback` | `Callable[[str], None]` | Fonction optionnelle pour le streaming | ❌ Non |

### GROUND_TRUTH et FEEDBACK 

#### Ground Truth (Vérité de terrain)

**Définition** : La `ground_truth` est la réponse correcte, attendue, ou de référence pour une question donnée.

**Utilisation** :
- Permet au Reflector de comparer la réponse du Generator avec la réponse correcte
- Facilite l'identification précise des erreurs
- Permet un apprentissage supervisé : le système apprend de ses erreurs en comparant avec la vérité

**Exemple** :
```python
question = "Quels sont les 5 critères d'un true sale ?"
ground_truth = """
1. Le transfert d'actifs doit être absolu et non comme garantie d'un prêt
2. L'originator doit renoncer au contrôle sur les actifs
3. Le SPV doit assumer les risques et récompenses de propriété
4. La vente doit être écrite et enregistrée
5. La vente doit être à distance et à juste valeur
"""
```

**Dans le prompt du Reflector** :
Quand `ground_truth` est fourni, le message inclut :
```
GROUND TRUTH / EXPECTED ANSWER:
[contenu de ground_truth]
```

Le Reflector peut alors :
- Identifier exactement ce qui manque dans la réponse du Generator
- Détecter les informations incorrectes
- Comprendre pourquoi la réponse était incomplète ou erronée

#### Feedback (Retour humain)


**Utilisation** :
- Fournit un contexte supplémentaire que la ground_truth seule ne capture pas
- Peut indiquer des problèmes de style, de clarté, ou de formatage
- Utile pour l'apprentissage par renforcement humain (RLHF)

**Types de feedback possibles** :
1. **Feedback positif** : "Excellente réponse, très claire et complète"
2. **Feedback négatif** : "La réponse manque de précision sur le point 3"
3. **Feedback constructif** : "Ajoutez plus de détails sur les conséquences légales"
4. **Feedback de style** : "Le ton est trop technique, simplifiez pour un public non-expert"

**Exemple** :
```python
feedback = """
La réponse est globalement correcte mais :
- Manque de précision sur le critère de renonciation de contrôle
- Ne mentionne pas les conséquences en cas de non-respect
- Le formatage pourrait être amélioré avec des sous-points
"""
```

**Dans le prompt du Reflector** :
Quand `feedback` est fourni, le message inclut :
```
HUMAN FEEDBACK:
[contenu du feedback]
```

Le Reflector peut alors :
- Comprendre les attentes humaines au-delà de la simple exactitude
- Identifier les problèmes de présentation ou de style
- Extraire des insights sur ce qui rend une réponse "bonne" selon les humains


#### Scénarios d'utilisation

**Scénario 1 : Avec Ground Truth uniquement**
```python
result = pipeline.run(
    question="Qu'est-ce qu'un true sale ?",
    ground_truth="Un true sale est un transfert d'actifs qui..."
)
```
→ Le Reflector compare directement et identifie les écarts

**Scénario 2 : Avec Feedback uniquement**
```python
result = pipeline.run(
    question="Rédigez une clause de true sale",
    feedback="La clause est bonne mais manque de détails sur les garanties"
)
```
→ Le Reflector comprend les attentes qualitatives sans réponse de référence

**Scénario 3 : Avec les deux**
```python
result = pipeline.run(
    question="Expliquez le credit enhancement",
    ground_truth="Le credit enhancement est...",
    feedback="Excellent, mais ajoutez des exemples concrets"
)
```
→ Le Reflector a à la fois la référence correcte et les attentes qualitatives

**Scénario 4 : Sans aucun des deux**
```python
result = pipeline.run(
    question="Qu'est-ce qu'un SPV ?"
)
```
→ Le Reflector analyse uniquement basé sur sa propre expertise et le playbook

### Processus interne

1. **Formatage des entrées** : Le playbook est formaté, et le message utilisateur est construit avec :
   - La question originale
   - La sortie du Generator (en JSON)
   - Le playbook actuel
   - La ground_truth (si fournie)
   - Le feedback (si fourni)

2. **Analyse par le LLM** : Le Reflector utilise un prompt système qui lui demande de :
   - Analyser le raisonnement du Generator
   - Identifier les erreurs conceptuelles, les définitions mal appliquées, les étapes manquantes
   - Déterminer la cause racine de chaque erreur
   - Suggérer l'approche correcte
   - Extraire les insights clés
   - Tagger chaque bullet utilisé comme "helpful", "harmful", ou "neutral"

3. **Parsing de la réponse** : La réponse JSON est parsée pour créer un objet `ReflectorOutput`

### OUTPUTS (Sorties) - ReflectorOutput

| Champ | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Analyse détaillée de la réponse du Generator, expliquant ce qui a bien fonctionné et ce qui n'a pas fonctionné |
| `error_identification` | `str` | Description précise de ce qui a mal fonctionné dans la réponse |
| `root_cause_analysis` | `str` | Explication de la cause profonde de l'erreur (ex: mauvaise source, terme mal interprété, bullet non appliqué) |
| `correct_approach` | `str` | Description de l'approche correcte que le Generator devrait utiliser la prochaine fois |
| `key_insight` | `str` | Le principe ou la leçon clé à retenir pour améliorer les générations futures |
| `bullet_tags` | `List[Dict[str, str]]` | Liste de tags pour chaque bullet utilisé, format : `[{"id": "str-00001", "tag": "helpful"}]`. Les tags peuvent être "helpful", "harmful", ou "neutral" |
| `raw_response` | `Optional[LLMResponse]` | Réponse brute du LLM (pour debugging) |

### Exemple de sortie

```json
{
    "reasoning": "Le Generator a correctement identifié les 5 critères du true sale mais a omis d'expliquer les conséquences légales en cas de non-respect. La réponse était factuellement correcte mais incomplète.",
    "error_identification": "Manque d'explication sur les conséquences de la non-conformité aux critères de true sale",
    "root_cause_analysis": "Le bullet str-00001 liste les critères mais ne mentionne pas les conséquences. Le Generator n'a pas utilisé de connaissances générales pour compléter.",
    "correct_approach": "Le Generator devrait : 1) Lister les critères (fait) ; 2) Expliquer brièvement chaque critère ; 3) Mentionner les conséquences légales (recharacterization risk, perte de bankruptcy remoteness)",
    "key_insight": "Lorsqu'on liste des critères légaux, toujours inclure les conséquences de leur non-respect pour donner un contexte complet",
    "bullet_tags": [
        {"id": "str-00001", "tag": "helpful"},
        {"id": "def-00001", "tag": "neutral"}
    ]
}
```

### Mode avec raffinement itératif

Le Reflector peut aussi fonctionner en mode `reflect_with_refinement()` qui effectue plusieurs passes :
- **Itération 1** : Analyse initiale
- **Itérations 2-N** : Raffinement basé sur l'analyse précédente
- Permet d'améliorer la qualité des insights extraits

---

## AGENT 3 : CURATOR (Curateur)

### Rôle
Le Curator est responsable de maintenir et d'enrichir le playbook. Il décide quelles nouvelles connaissances doivent être ajoutées au playbook basé sur les insights du Reflector.

### INPUTS (Entrées)

| Paramètre | Type | Description | Obligatoire |
|-----------|------|-------------|-------------|
| `question` | `str` | La question originale | ✅ Oui |
| `generator_output` | `GeneratorOutput` | La sortie du Generator | ✅ Oui |
| `reflector_output` | `ReflectorOutput` | La sortie du Reflector contenant les insights | ✅ Oui |
| `playbook` | `Playbook` | Le playbook actuel | ✅ Oui |
| `stream_callback` | `Callable[[str], None]` | Fonction optionnelle pour le streaming | ❌ Non |

### Processus interne

1. **Examen des insights** : Le Curator examine :
   - Les insights clés identifiés par le Reflector (`key_insight`)
   - L'analyse des erreurs (`error_identification`, `root_cause_analysis`)
   - L'approche correcte suggérée (`correct_approach`)

2. **Vérification de redondance** : Le Curator compare avec le contenu existant du playbook pour éviter :
   - Les doublons exacts
   - Les contenus très similaires
   - Les informations déjà couvertes

3. **Décision d'ajout** : Pour chaque insight, le Curator détermine :
   - Si l'information est nouvelle et utile
   - Dans quelle section l'ajouter (strategies, pitfalls, templates, definitions)
   - Comment formuler le contenu de manière concise et actionnable

4. **Génération des opérations** : Le Curator produit une liste d'opérations à effectuer

### OUTPUTS (Sorties) - CuratorOutput

| Champ | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Explication de pourquoi ces ajouts sont nécessaires et comment ils améliorent le playbook |
| `operations` | `List[Dict[str, Any]]` | Liste des opérations à effectuer. Chaque opération contient :<br>- `type` : "ADD" (pour l'instant, seul type supporté)<br>- `section` : Une des sections du playbook ("strategies", "pitfalls", "templates", "definitions")<br>- `content` : Le texte du nouveau bullet à ajouter |
| `raw_response` | `Optional[LLMResponse]` | Réponse brute du LLM (pour debugging) |

### Exemple de sortie

```json
{
    "reasoning": "Le Reflector a identifié un insight important : lors de l'explication de critères légaux, il faut toujours inclure les conséquences. Ce principe n'existe pas encore dans le playbook et devrait être ajouté comme stratégie.",
    "operations": [
        {
            "type": "ADD",
            "section": "strategies",
            "content": "Lorsqu'on explique des critères légaux ou des exigences réglementaires, toujours inclure les conséquences de leur non-respect (pénalités, risques de recharacterization, perte de protections) pour fournir un contexte complet et actionnable."
        }
    ]
}
```

### Méthode apply_updates()

Après que le Curator ait produit ses opérations, la méthode `apply_updates()` est appelée :

**Processus** :
1. **Application des tags de bullets** : Les tags du Reflector (`bullet_tags`) sont appliqués
   - Chaque bullet tagué comme "helpful" voit son `helpful_count` incrémenté
   - Chaque bullet tagué comme "harmful" voit son `harmful_count` incrémenté
   - Chaque bullet tagué comme "neutral" voit son `neutral_count` incrémenté

2. **Application des opérations** : Les opérations du Curator sont exécutées
   - Pour chaque opération de type "ADD", un nouveau bullet est créé dans la section appropriée
   - Un ID unique est généré (ex: `str-00005`)
   - Le bullet est ajouté à la section correspondante

3. **Retour** : La méthode retourne la liste des nouveaux bullets ajoutés

**Exemple** :
```python
added_bullets = curator.apply_updates(curator_output, reflector_output)
# Retourne : [Bullet(id="str-00005", content="Lorsqu'on explique...", ...)]
```


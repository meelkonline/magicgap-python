from transformers import pipeline

req = ""

messages = [
    {"role": "user", "content": """
###
User Question : Comment confirmer ma commande?
###
- You are Sophia, a real estate expert
- The conversation takes place during a real estate training session, in France.
- You answer questions from a learner.
- You are named Sophia.
- Current date and time: 2024-06-25 16:07:43. 
###

###
Your Role: 
- Respond accurately and relevantly to the user's questions about any technical or internal subjects.
- Your answers should be concise, within 40 words or less.
- Answer in French only.
- use CAPITAL LETTERS to emphasize meaningful word in your answer.
- use ... (three points) to indicate a pause in your answer, when relevant.

Please find relevant chunks for this question
###
CONDITIONS GENERALES DE VENTE AVENIR & TALENT
<<<
Par ailleurs, la facturation sera automatiquement déclenchée au jour de la commande. Par voie de conséquence, ils seront majorés du montant de la TVA au taux légal en vigueur au moment de la facturation. Cette facture sera établie en fin d’année et payable à réception. En sus des indemnités de retard, toute somme non payée dans un délai de 30 jours à compter de son exigibilité produira de plein droit le paiement d’une indemnité forfaitaire de 40 euros due au titre des frais de recouvrement.
>>>
<<<
9. RETARD DE PAIEMENT En cas de défaut de paiement total ou partiel des formations fournies à échéance, le Client doit verser à AVENIR & TALENT une pénalité de retard égale au taux d’intérêt appliqué par la Banque Centrale Européenne à son opération de refinancement la plus récente majoré de 10 (dix) points. Cette pénalité est calculée sur le montant TTC de la somme restant due, et court à compter de la date d’échéance du prix sans qu’aucune mise en demeure préalable ne soit nécessaire.
>>>
<<<
Le Client ne pourra, sans aucun prétexte, en obtenir le remboursement. En cas d’annulation par AVENIR & TALENT, le Client ne pourra réclamer aucun dédommagement, réparation ou dédit. Celle-ci est récupérable par le Client et non-imputable sur sa participation légale à la Formation Professionnelle Continue. Aucun escompte ne sera consenti en cas de paiement anticipé.
>>>
<<<
le 6ème jour ouvré du début de la formation, le Client n’a plus la possibilité de le faire par le biais de son compte Plateforme F. La demande devra impérativement être faite auprès de AVENIR & TALENT, à l’adresse susmentionnée. En cas d’annulation reçue moins de 6 jours ouvrés avant la date du premier jour de la formation, AVENIR & TALENT se réserve le droit de facturer au Client, 100% du coût de la formation TTC à titre d’indemnité. AVENIR & TALENT se réserve le droit, sans indemnité de quelque nature que ce soit, de refuser toute inscription ou accès à un Client qui ne serait pas à jour de ses paiements, d’exclure tout participant qui aurait procédé à de fausses déclarations lors de l’inscription, ainsi que tout participant dont le comportement gênerait le bon déroulement de la formation (présentiel ou en visio-conférence) ou manquerait au règlement intérieur de AVENIR & TALENT. Les participants aux formations et le Client peuvent exercer leur droit d’accès, d’’interrogation, d’opposition, de communication, de rectification ou de suppression de leurs données par courriel à l’adresse suivante :
>>>
<<<
3. ANNULATION ET REPORT DU CLIENT
Est considérée comme annulation toute Prestation annulée à une date prévue initialement et qui ne serait pas replanifiée dans les mois suivants la planification initiale. Est considérée comme report toute Prestation annulée à une date prévue initialement et qui serait planifiée de nouveau dans les mois suivants la planification initiale. Il est précisé que pour toute annulation ou report entre le 15ème jour ouvré et avant Les parties devront mettre en oeuvre tous leurs efforts pour prévenir ou réduire les effets d’une inexécution de la Prestation causée par un événement de force majeure ; la partie désirant invoquer un événement de force majeure devra notifier immédiatement à l’autre partie le commencement et la fin de cet événement, sans quoi elle ne pourra être déchargée de sa responsabilité.
>>>
<<<
Ils sont libellés en euros et calculés hors taxes.
>>>
<<<
Pour toute formation à laquelle le participant ne s’est pas présenté ou n’a assisté que partiellement, les frais de formation seront facturés à 100% à titre d’indemnité. AVENIR & TALENT avancera pour le compte du Client l’ensemble des coûts pédagogiques ainsi que les frais de restauration (midi) pour chaque jour de formation. En cas d’octroi de subvention par l’OPCO, cette subvention sera imputée sur le prix de la formation.
>>>
<<<
La responsabilité de AVENIR & TALENT est expressément limitée à l’indemnisation des dommages directs prouvés par le Client.
>>>
<<<
FORCE MAJEURE Les obligations contenues aux présentes ne seront pas applicables ou seront suspendues si leur exécution est devenue impossible en raison d’un cas de force majeure tels que notamment : acte de puissance publique, hostilités, guerre, fait du Prince, catastrophe naturelle, incendie, inondation.
>>>
<<<
Enfin, le Client devra cliquer sur la case « Je confirme ma commande ». Le Client devra s’acquitter du reste à charge.
>>>"""},
]

from transformers import pipeline

pipe = pipeline("text-generation", model="Nekochu/Llama-2-13B-fp16-french")
output = pipe(messages)
print(output)

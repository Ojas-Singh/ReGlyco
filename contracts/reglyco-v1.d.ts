export type ReGlycoProfile = 'public' | 'full';
export type WorkflowId = 'uniprot' | 'n_scan' | 'site_build' | 'ensemble' | 'relax' | 'validate' | 'refine' | 'density' | 'saxs';
export interface SiteKey { model: number; chain: string; residueNumber: number; insertionCode?: string | null }
export interface SiteAssignment {
  id: string; site: SiteKey; residueName: string; glycanId: string;
  anomer: 'alpha' | 'beta' | 'unknown'; level: 1 | 2 | 3;
  provenance: 'uniprot' | 'scan' | 'manual'; sourceDescription?: string;
  evidence?: string; glycanAsset?: string; excluded?: boolean; unresolved?: boolean;
  /** Explicitly replace the deposited glycan at this attachment site. */
  replaceExisting?: boolean;
  /** Identifier for the deposited glycan being replaced, when known. */
  existingGlycanId?: string | null;
  /** Residues in the deposited glycan tree, used as a parser-independent fallback. */
  existingGlycanResidues?: SiteKey[];
}
export interface ProgressEvent { stage: string; message: string; current?: number; total?: number; fraction?: number }
export interface ValidationFinding {
  code: string; severity: 'info' | 'warning' | 'error'; message: string;
  site?: SiteKey; observed?: number; expected?: string;
  domain?: string; origin?: 'inherited from input' | 'inherited from glycan asset' | 'introduced' | 'resolved' | string;
  stage?: string; frame?: number; glycanIndex?: number; glycan?: string; linkage?: string;
  involvedAtoms?: string[]; metric?: string; policy?: string; policyVersion?: string;
}
export interface StericPolicy {
  prescreenDistanceAngstrom: number; clearOverlapAngstrom: number;
  hardOverlapAngstrom: number; model: string;
}
export interface StericContact {
  first: number; second: number; distanceAngstrom: number; overlapAngstrom: number;
  class: 'clear' | 'advisory' | 'hard';
}
export interface StericSummary {
  contacts: StericContact[]; clearCount: number; advisoryCount: number; hardCount: number;
  maximumOverlapAngstrom: number; totalOverlapAngstrom: number;
}
export interface StrictSearchSiteDiagnostic {
  site: SiteKey; phiDegrees: number; psiDegrees: number;
  phiComponent: number; psiComponent: number;
  phiWithin95: boolean; psiWithin95: boolean;
  phiLower95Degrees?: number; phiUpper95Degrees?: number;
  psiLower95Degrees?: number; psiUpper95Degrees?: number;
  stericScore: number;
}
export interface StrictSearchDiagnostics {
  generation: number; frozenSites: number; evaluations: number; repaired: boolean;
  sites: StrictSearchSiteDiagnostic[]; outlierSites?: SiteKey[]; vdwOutlierSites?: SiteKey[];
  vdwHardContacts?: number; vdwAdvisoryContacts?: number;
  vdwMaxOverlapAngstrom?: number; vdwTotalOverlapAngstrom?: number;
  vdwContacts?: string[]; bestCandidatePdb: string;
}
export interface ValidationSummary {
  valid: boolean; findings: ValidationFinding[]; warnings: string[]; errors: string[];
  componentDictionaryVersion?: string; stericPolicy?: StericPolicy; stericSummary?: StericSummary;
}
export type TorsionAssessment = 'core' | 'allowed' | 'tail' | 'outlier' | 'no_reference';
export interface TorsionObservation {
  domain: string; origin: string; stage: string; frame?: number;
  site?: string; branchPath?: string[]; branch_path?: string[];
  glycanIndex: number; linkage: string; involvedAtoms?: string[];
  phiDegrees?: number; psiDegrees?: number; omegaDegrees?: number;
  populationPercentile?: number; assessment: TorsionAssessment;
  nearestPopulation?: number; selectedPopulation?: number; policyVersion?: string;
}
export interface AttachmentTorsionObservation {
  domain?: string; origin?: string; site: string; stage: string; frame?: number; phiDegrees: number; psiDegrees: number;
  glycanIndex?: number; linkage?: string; involvedAtoms?: string[];
  populationPercentile?: number; assessment: TorsionAssessment;
  selectedPhiComponent?: number; selectedPsiComponent?: number; policyVersion?: string;
}
export interface SearchDiagnostics {
  selectionPolicy?: string; priorModel?: string; description?: string;
  populationSize?: number; generationLimit?: number; searchBudget?: number;
  evaluations?: number; validCandidates?: number;
  firstFeasibleScore?: number | null; finalPriorScore?: number | null;
  terminationReason?: string;
  sites?: Array<{ site?: unknown; conformerId?: string; conformerProbability?: number | null;
    attachmentLogDensity?: number | null; jointPriorScore?: number | null;
    phiDegrees?: number; psiDegrees?: number; stericScore?: number; populationSource?: string }>;
  populationSources?: string[];
}
export interface TorsionPopulation {
  index: number; level?: number; parentIndex?: number; color?: string; weight?: number;
  phiMean?: number; psiMean?: number; omegaMean?: number;
  phiKappa?: number; psiKappa?: number; omegaKappa?: number;
  medoidPhi?: number; medoidPsi?: number; medoidOmega?: number;
  sourceCluster?: number;
}
export interface TorsionContour {
  mass?: number; bins?: number; grid?: number[]; counts?: number[]; threshold?: number;
}
export interface TorsionClusterGrid {
  id?: number; pct?: number; color?: string;
  phiPsi?: TorsionContour; phiOmega?: TorsionContour; psiOmega?: TorsionContour;
  phiHist?: number[]; psiHist?: number[]; omegaHist?: number[];
}
export interface TorsionReference {
  version?: string; sourceDatabase?: string; source_database?: string; contentHash?: string; content_hash?: string;
  canonicalLinkage?: string; canonical_linkage?: string; branchPath?: string[]; branch_path?: unknown[];
  phiPsi?: TorsionContour; phiOmega?: TorsionContour; psiOmega?: TorsionContour;
  phiOmegaContours?: TorsionContour[]; phi_omega_contours?: TorsionContour[];
  psiOmegaContours?: TorsionContour[]; psi_omega_contours?: TorsionContour[];
  populations?: TorsionPopulation[]; contours?: TorsionContour[]; clusters?: TorsionClusterGrid[];
}
export interface AttachmentVmmComponent {
  index: number; meanDegrees: number; concentration: number; weight: number;
  lower95Degrees?: number; upper95Degrees?: number;
}
export interface AttachmentReference {
  site: string | SiteKey; phi: AttachmentVmmComponent[]; psi: AttachmentVmmComponent[];
  selectedPhiComponent?: number; selectedPsiComponent?: number;
  phiWithinVmm95?: boolean; psiWithinVmm95?: boolean;
}
export interface EnsembleTorsionSummary {
  frames: number; observedFrames?: number[]; outlierFrames?: number[];
  linkageOutlierRates?: Record<string, number>;
  circularMeans?: Record<string, [number, number]>;
  circularDispersion?: Record<string, [number, number]>;
  clusterCoverage?: Record<string, number>;
}
export interface TorsionClusterComparison {
  index: number; color?: string; referenceFraction: number; observedFraction: number;
  observedFrames: number; differencePercentagePoints: number;
}
export interface TorsionClusterDistribution {
  key: string; site?: string; glycanIndex?: number; branchPath: string[]; linkage: string;
  frames: number; unassignedFrames: number; clusters: TorsionClusterComparison[];
}
export interface ReportAnalysis {
  torsionReferences?: TorsionReference[]; torsionObservations?: TorsionObservation[];
  attachmentReferences?: AttachmentReference[]; attachmentObservations?: AttachmentTorsionObservation[];
  ensembleTorsion?: EnsembleTorsionSummary; clusterDistributions?: TorsionClusterDistribution[];
  searchDiagnostics?: SearchDiagnostics;
  [key: string]: unknown;
}
export interface ReGlycoOptions {
  ensembleMode?: "sampled" | "conformer_collection";
  ensembleBurnInSteps?: number;
  ensembleThinningSteps?: number;
  computeBackend?: 'auto' | 'cpu' | 'webgpu';
  /** Numerical target for sampled energy/interaction ensembles. */
  samplingTarget?: 'cpu_reference_v1' | 'webgpu_f32_v1';
  preMinimization?: boolean;
  /** Optional in the editor; submitted Build/Ensemble requests contain the effective seed. */
  seed?: number;
  /** Provider residue-name convention for Build/Ensemble; omitted legacy requests mean PDB. */
  outputFormat?: 'PDB' | 'GLYCAM';
  postRelax: boolean; scanRotamers: boolean;
  ensembleFrames: number; ensembleTemperatureK: number;
  ensembleMhChains: number; ensembleBurnInSweeps: number;
  ensembleThinningAccepted: number; calculateSasa: boolean;
  calculateHotspots: boolean; scoringMode: 'steric_prior' | 'full_energy' | 'protein_glycan_interaction';
  useObc2: boolean; localRadius: number; populationSize: number; generations: number;
  [key: string]: unknown;
}
export interface ReglycoRunRequestV1 {
  schemaVersion: 1; workflow: WorkflowId; profile: ReGlycoProfile;
  input: { kind: 'uniprot' | 'pdb_id' | 'upload' | 'job'; label: string; sourceId?: string; sourceUrl?: string; asset: string; sha256: string };
  assignments: SiteAssignment[]; options: ReGlycoOptions; parentJobId?: string; createdAt: string;
}
export interface WorkflowArtifact { name: string; mediaType: string; role: 'input' | 'structure' | 'analysis' | 'report' | 'provenance' | 'other'; data: string | Uint8Array }
export interface WorkflowReport {
  workflow: WorkflowId; title: string; summary: string; generatedAt: string;
  engineVersion: string; inputSha256: string;
  sites: Array<{ site: SiteKey; glycanId?: string; status: string; score?: number; details?: Record<string, unknown> }>;
  validation: ValidationSummary; analysis: ReportAnalysis; provenance: Record<string, unknown>;
}
export interface WorkflowBundle {
  schemaVersion: 1; status: 'succeeded' | 'partial' | 'failed' | 'cancelled'; workflow: WorkflowId;
  primaryStructure?: string; report: WorkflowReport; artifacts: WorkflowArtifact[]; warnings: string[]; error?: string;
}

// Barrido: % de veces que la red ENTRENADA generaliza el patron XOR reservado,
// y % de veces que el TRN lo acierta. Con 40 runs -> resultado estadistico honesto.
use rand::{SeedableRng, rngs::StdRng};
use trn_benchmark::trn::TRN;
use trn_benchmark::tiny_nn::TinyNN;

const D: usize = 800;
const W: usize = 50;
const K: usize = 2;

fn main() {
    let xors: Vec<([usize;2], usize)> = vec![([0,0],0),([0,1],1),([1,0],1),([1,1],0)];
    let train_idx = [0usize,1usize,2usize];
    let reserved = 3usize;
    let runs = 60;
    let mut trn_hits=0; let mut nn_hits=0;

    for seed in 0..runs {
        let mut rng = StdRng::seed_from_u64(seed as u64 ^ 0x9e37);

        // TRN
        let mut trn = TRN::new_v2(D,W,K);
        for _ in 0..400 { for &i in &train_idx { let (inp,_)=xors[i]; trn.train_sequence(&vec![format!("x{}",inp[0]),format!("y{}",inp[1])],&mut rng); } }
        let (rinp,rtarget)=xors[reserved];
        let c = trn.predict(&vec![format!("x{}",rinp[0]),format!("y{}",rinp[1])],&mut rng).map(|(_,c)|c).unwrap_or(0.0);
        let rpred = if c>0.5 {1}else{0};
        if rpred==rtarget { trn_hits+=1; }

        // NN
        let mut nn = TinyNN::new(2,16,2,&mut rng);
        for _ in 0..1500 { for &i in &train_idx { let (inp,t)=xors[i]; nn.train(&[inp[0] as f32,inp[1] as f32],t,0.02); } }
        let p = nn.forward(&[rinp[0] as f32,rinp[1] as f32]);
        let npred = if p[1]>p[0] {1}else{0};
        if npred==rtarget { nn_hits+=1; }
    }
    println!("Barrido {} runs: generalizacion sobre el patron XOR reservado (1,1)->0", runs);
    println!("  TRN  reservado acertado: {}/{} = {:.0}%  (memoriza los vistos, azar en el nuevo)", trn_hits, runs, 100.0*trn_hits as f64/runs as f64);
    println!("  NN   reservado acertado: {}/{} = {:.0}%  (si >50% aprendio la funcion, no solo memorizo)", nn_hits, runs, 100.0*nn_hits as f64/runs as f64);
    println!("(azar = 50%, pues 2 clases)");
}

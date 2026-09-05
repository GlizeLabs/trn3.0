// Diagnostico: ¿ambos modelos colapsan a una clase fija?
use rand::{SeedableRng, rngs::StdRng};
use trn_benchmark::trn::TRN;
use trn_benchmark::tiny_nn::TinyNN;

const D: usize = 800;
const W: usize = 50;
const K: usize = 2;

fn main() {
    let xors: Vec<([usize;2], usize)> = vec![([0,0],0),([0,1],1),([1,0],1),([1,1],0)];
    let train_idx = [0usize,1usize,2usize];
    let mut rng = StdRng::seed_from_u64(1);

    // TRN: que devuelve para cada patron?
    let mut trn = TRN::new_v2(D,W,K);
    for _ in 0..400 { for &i in &train_idx { let (inp,_)=xors[i]; trn.train_sequence(&vec![format!("x{}",inp[0]),format!("y{}",inp[1])],&mut rng); } }
    println!("TRN confidences por patron:");
    for &(inp,target) in &xors {
        let c = trn.predict(&vec![format!("x{}",inp[0]),format!("y{}",inp[1])],&mut rng).map(|(pieza,conf)| conf).unwrap_or(0.0);
        println!("  {:?} -> esperado {} | conf {:.4}", inp, target, c);
    }

    // NN: probs por patron
    let mut nn = TinyNN::new(2,16,2,&mut rng);
    for _ in 0..1500 { for &i in &train_idx { let (inp,t)=xors[i]; nn.train(&[inp[0] as f32,inp[1] as f32],t,0.02); } }
    println!("NN probs por patron:");
    for &(inp,target) in &xors {
        let p = nn.forward(&[inp[0] as f32,inp[1] as f32]);
        println!("  {:?} -> esperado {} | P(clase0)={:.3} P(clase1)={:.3} pred {}",
            inp, target, p[0], p[1], if p[1]>p[0]{1}else{0});
    }
}

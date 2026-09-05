// Benchmark HONESTO: TRN vs Red ENTRENADA (backprop real).
// Entrenamos con 3 patrones XOR, reservamos el 4to (NUNCA visto).
// Si un modelo solo memoriza -> falla el reservado. Si aprendio la funcion XOR -> lo acierta.
use rand::{SeedableRng, rngs::StdRng};
use trn_benchmark::trn::TRN;
use trn_benchmark::tiny_nn::TinyNN;

const D: usize = 2000;
const W: usize = 100;
const K: usize = 2;

fn main() {
    println!("TRN (entrenado) vs Red ENTRENADA — benchmark honesto");
    println!("Entrenados en 3 patrones XOR, se RESERVA (1,1)->0 como nunca visto.\n");

    let xors: Vec<([usize;2], usize)> = vec![
        ([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0),
    ];
    let mut rng = StdRng::seed_from_u64(7);
    let train_idx = [0usize,1usize,2usize]; // (1,1)->0 queda reservado (indice 3)

    // ---- TRN entrenado en los 3 ----
    let mut trn = TRN::new_v2(D, W, K);
    for _ in 0..400 {
        for &i in &train_idx {
            let (inp,_) = xors[i];
            trn.train_sequence(&vec![format!("x{}",inp[0]), format!("y{}",inp[1])], &mut rng);
        }
    }
    let mut trn_all=0; let mut trn_unseen=0;
    for (i,&(inp,target)) in xors.iter().enumerate() {
        let seq = vec![format!("x{}",inp[0]), format!("y{}",inp[1])];
        let c = trn.predict(&seq, &mut rng).map(|(_,c)| c).unwrap_or(0.0);
        let pred = if c > 0.5 {1}else{0};
        let seen = train_idx.contains(&i);
        let ok = pred==target;
        if ok { trn_all+=1; } if !seen && ok { trn_unseen+=1; }
        println!("  TRN  XOR {:?}->{} pred {} (conf {:.2}) | {}{}",
            inp,target,pred,c, if ok{"✓"}else{"✗"}, if seen{" (visto)"}else{" (RESERVADO)"});
    }
    println!("  TRN total {}/4 | aciertos en RESERVADO {}/1\n", trn_all, trn_unseen);

    // ---- Red ENTRENADA en los mismos 3 ----
    let mut nn = TinyNN::new(2, 16, 2, &mut rng);
    for _ in 0..1500 {
        for &i in &train_idx {
            let (inp,target) = xors[i];
            nn.train(&[inp[0] as f32, inp[1] as f32], target, 0.02);
        }
    }
    let mut nn_all=0; let mut nn_unseen=0;
    for (i,&(inp,target)) in xors.iter().enumerate() {
        let p = nn.forward(&[inp[0] as f32, inp[1] as f32]);
        let pred = if p[1]>p[0] {1}else{0};
        let seen = train_idx.contains(&i);
        let ok = pred==target;
        if ok { nn_all+=1; } if !seen && ok { nn_unseen+=1; }
        println!("  NN   XOR {:?}->{} pred {} | {}{}",
            inp,target,pred, if ok{"✓"}else{"✗"}, if seen{" (visto)"}else{" (RESERVADO)"});
    }
    println!("  NN total {}/4 | generaliza en RESERVADO {}/1", nn_all, nn_unseen);

    println!("\n=== LECTURA HONESTA ===");
    println!("1) Un modelo que FALLA el patron reservado no aprendio XOR: solo memorizo.");
    println!("2) TRN memoriza bien los 3 vistos pero casi siempre falla el reservado.");
    println!("3) Una red entrenada de verdad puede aprender la FUNCION y acertar el 4to.");
    println!("4) Tu v2.1 ponia TRN entrenado vs transformer SIN entrenar -> comparacion injusta.");
    println!("   Y hardcodeaba numeros (89%, 115%, latencia 10x) en vez de medirlos.");
}

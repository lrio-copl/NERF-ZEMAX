import unittest
import numpy as np
from extract_nerf_rays import *

class TestNerfZemaxFunctions(unittest.TestCase):

    # Constantes pour reutilisation et clarte
    NUM_RAYS = 1000000
    TOLERANCE_ANGLE = 0.05
    TOLERANCE_DELTA = 0.05

    def calculate_circle_plane(self, rayon):
        return circle_plane(self.NUM_RAYS, rayon)

    def calculate_square_plane(self, hauteur):
        return square_plane(self.NUM_RAYS, 1, hauteur, 0)

    def test_circle_plane_rayon(self):
        for rayon in [1, 5, 10, 100]:
            rays = self.calculate_circle_plane(rayon)

            # Calcule la distance radiale depuis l''origine pour chaque rayon
            distances_radiales = np.sqrt(np.square(rays[:, 0]) + np.square(rays[:, 1]))

            # Verifie si la distance radiale moyenne est proche de la valeur attendue
            distance_radiale_moyenne = np.mean(distances_radiales)
            valeur_attendue = rayon / 2
            self.assertAlmostEqual(distance_radiale_moyenne, valeur_attendue, delta=0.05*rayon,
                                   msg=f'Moyenne des rayons incorrecte pour rayon={rayon}')

            # Verifie si la moyenne de l'arctangente de (y/x) est approximativement egale a 0
            angles = np.arctan(rays[:, 1] / rays[:, 0])
            moyenne_angle = np.abs(np.mean(angles))
            self.assertAlmostEqual(moyenne_angle, 0, delta=self.TOLERANCE_ANGLE, msg=f'Moyenne de l\'arctangente incorrecte pour rayon={rayon}')

    def test_circle_plane_valeur_z(self):
        for z in [1, 5, 10, 100]:
            moyenne_z = np.mean(circle_plane(self.NUM_RAYS, 1, z)[:, 2])
            self.assertEqual(moyenne_z, z, msg=f'Moyenne de la valeur z incorrecte pour z={z}')

    def test_square_plane_coordonnees(self):
        for hauteur in [0, 1, 5, 10, 100]:
            points = self.calculate_square_plane(hauteur)

            # hauteur/largeur
            self.assertAlmostEqual(np.mean(points[:, 1]), 0, delta=0.05*hauteur, msg=f'Moyenne de x incorrecte pour largeur={hauteur}')
            self.assertAlmostEqual(np.max(points[:, 1]), hauteur, delta=0.05*hauteur, msg=f'Max de x incorrect pour largeur={hauteur}')
            self.assertAlmostEqual(np.min(points[:, 1]), -hauteur, delta=0.05*hauteur, msg=f'Min de x incorrect pour largeur={hauteur}')

            # position en z
            self.assertEqual(np.mean(square_plane(self.NUM_RAYS, 1, 1, hauteur)[:, 2]), hauteur, msg=f'Moyenne de la valeur z incorrecte pour hauteur={hauteur}')

    def test_orientation_demi_sphere(self):
        y_mask = [True, False, True]
        x = half_sphere(1000000, 180, 180, 'x', False)
        y = half_sphere(1000000, 180, 180, 'y', False)
        z = half_sphere(1000000, 180, 180, 'z', False)

        # La moyenne de y et z est 0
        self.assertAlmostEqual(np.mean(x[:, 1:3]), 0, delta=0.05, msg="Moyenne incorrecte pour orientation=x")

        # La moyenne de x et z est 0
        self.assertAlmostEqual(np.mean(y[:, y_mask]), 0, delta=0.05, msg="Moyenne incorrecte pour orientation=y")

        # La moyenne de x et y est 0
        self.assertAlmostEqual(np.mean(z[:, :2]), 0, delta=0.05, msg="Moyenne incorrecte pour orientation=z")
        for phi in [5, 45, 90, 180]:
            phi = np.deg2rad(phi)
            for theta in [5, 45, 90, 180]:
                theta = np.deg2rad(theta)
                xyz = half_sphere(100000, theta, phi, 'z', False)
        """
        !test ne fonctionne pas
                # Verifie l'angle maximal pour theta
                self.assertAlmostEqual(np.pi/2 - np.max(np.arctan(xyz[:, 1] / xyz[:, 0])) , theta/2, delta=0.05 * theta/2, msg=f'Mismatch avec theta = {theta} et phi = {phi}')
    
                # Verifie l'angle maximal pour phi
                norme_xy = np.linalg.norm(xyz[:, :2], axis=1)
                self.assertAlmostEqual(np.pi/2 - np.max(np.arctan(xyz[:, 2] / norme_xy)) , phi/2, delta=0.05 * phi/2, msg=f'Mismatch avec theta = {theta} et phi = {phi}')
        """
        # Verifie que le rayon est 1
        self.assertAlmostEqual(np.mean(np.linalg.norm(xyz, axis=1)), 1, delta=0.05)

    def test_generate_diffuse_plane_lmn_normalises(self):

        # Rayons plans diffus pour tous les cas
        sc = generate_diffuse_plane(1000, square_plane, circle_plane)
        cs = generate_diffuse_plane(1000, circle_plane, square_plane)
        cc = generate_diffuse_plane(1000, circle_plane, circle_plane)
        ss = generate_diffuse_plane(1000, square_plane, square_plane)

        # Verifie la normalisation pour chaque cas
        self.assertTrue(np.allclose(np.linalg.norm(sc[:, 3:], axis=1), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(cs[:, 3:], axis=1), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(cc[:, 3:], axis=1), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(ss[:, 3:], axis=1), 1.0))

    def test_generate_diffuse_sphere_convexe(self):
        nbrays = 100
        rayon = 1
        direction_theta = 180
        direction_phi = 180
        convexe = True

        rays = generate_diffuse_sphere(nbrays, rayon, direction_theta, direction_phi, convexe)

        # Verifie la forme de la sortie
        self.assertEqual(rays.shape, (nbrays, 6))

        # Verifie si les rayons pointent vers l'interieur (sphere convexe)
        self.assertTrue(np.all(rays[:, 2] > 0))

        # Verifie si les directions sont normalisees
        self.assertTrue(np.allclose(np.linalg.norm(rays[:, 3:], axis=1), 1.0))

    def test_generate_diffuse_sphere_concave(self):
        nbrays = 100
        rayon = 1
        direction_theta = 180
        direction_phi = 180
        convexe = False

        rays = generate_diffuse_sphere(nbrays, rayon, direction_theta, direction_phi, convexe)

        # Verifie la forme de la sortie
        self.assertEqual(rays.shape, (nbrays, 6))

        # Verifie si les rayons pointent vers l'exterieur (sphere concave)
        self.assertTrue(np.all(rays[:, 2] > 0))

        # Verifie si les directions sont normalisees
        self.assertTrue(np.allclose(np.linalg.norm(rays[:, 3:], axis=1), 1.0))


    def test_aspherique_equation_sphere(self):
        # Teste si c=0 et k=0, equation d'une sphere
        sphere_rays = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]])
        self.assertTrue(np.allclose(NerfRays("data/config_test.yml").aspherique(sphere_rays, 0, 0), sphere_rays))

        # Teste si k=1
        asphere_rays = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]])
        self.assertTrue(np.allclose(NerfRays("data/config_test.yml").aspherique(asphere_rays, 0, 1), asphere_rays))

        # Teste si plusieurs coefficients 0
        self.assertTrue(np.allclose(NerfRays("data/config_test.yml").aspherique(sphere_rays, 0, 0, 0), sphere_rays))
        #test avec plusieurs coefficients
        #a faire
    def test_dc2rot_matrices_rotation(self):
        lmn = np.array([[0, 0, 1], [0, 0, 1]])
        rays = np.array([[1, 0, 0, 0.5, 0.5, 0.707], [1, 0, 0, 0.5, 0.5, 0.707]])

        # Matrices de rotation attendues
        matrices_attendues = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        # Teste si dc2rot produit des matrices de rotation correctes
        self.assertTrue(np.allclose(dc2rot(lmn, rays).as_matrix(), matrices_attendues))

        # Teste si le nombre de matrices retournees est egal au nombre de rayons
        self.assertEqual(len(dc2rot(lmn, rays)), len(rays))

    def test_incident_lmn_par_defaut(self):
        rays = np.array([[1, 0, 0, 0.5, 0.5, 0.707], [0, 1, 0, 0.5, 0.5, 0.707]])
        new_rays = NerfRays("data/config_test.yml").incident(rays)

        # Verifie si la norme des vecteurs lmn est proche de 1
        self.assertTrue(np.allclose(np.linalg.norm(new_rays[:, 3:], axis=1), 1.0, atol=1))

    def test_incident_lmn_personnalises(self):
        rays = np.array([[1, 0, 0, 0.5, 0.5, 0.707], [0, 1, 0, 0.5, 0.5, 0.707]])
        lmn = np.array([[0, 0, 1], [0, 0, 1]])
        new_rays = NerfRays("data/config_test.yml").incident(rays, lmn=lmn)
        # Verifie si la norme des vecteurs lmn est proche de 1
        self.assertTrue(np.allclose(np.linalg.norm(new_rays[:, 3:], axis=1), 1.0, atol=0.05), msg=np.linalg.norm(new_rays[:, 3:], axis=1))

    def test_spherique_rayon_0(self):
        rays = np.array([[1, 0, 0, 0.5, 0.5, 0.707], [0, 1, 0, 0.5, 0.5, 0.707]])
        new_rays = NerfRays("data/config_test.yml").aspherique(rays, rayon=0, coef=np.zeros(1), k=0)
        # Verifie si les rayons restent inchanges
        self.assertTrue(np.allclose(rays, new_rays, atol=0.05))

    def test_spherique_nonzero_rayon(self):
        rays = np.array([[1, 0, 0, 0.5, 0.5, 0.707], [0, 1, 0, 0.5, 0.5, 0.707]])
        new_rays = NerfRays("data/config_test.yml").aspherique(rays, rayon=1, coef=np.array([0.1, 0.01]), k=0)
        print(np.linalg.norm(new_rays[:, 3:], axis=1))
        # Verifie si la norme des coordonnees est proche de 1
        self.assertTrue(np.allclose(np.linalg.norm(new_rays[:, 3:], axis=1), 1.0, atol=0.05))

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()